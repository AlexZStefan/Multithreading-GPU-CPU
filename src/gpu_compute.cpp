#include "gpu_compute.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

std::string ComputeGPU::loadShaderSource(const char* filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open shader file: ") + filePath);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void ComputeGPU::createTexture(int width, int height) {
    glGenTextures(1, &texColor);
    glBindTexture(GL_TEXTURE_2D, texColor);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width, height);
    glBindImageTexture(1, texColor, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}

GLuint ComputeGPU::createComputeShaderProgram(const char* shaderPath) {
    std::string src = loadShaderSource(shaderPath);
    const char* source = src.c_str();

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        throw std::runtime_error(std::string("Compute shader compilation failed: ") + log);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader program linking failed: ") + log);
    }

    glDeleteShader(shader);
    return program;
}

void ComputeGPU::init(const char* shaderPath) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    g_window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
    if (!g_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(g_window);

    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    g_program = createComputeShaderProgram(shaderPath);

    glGenBuffers(1, &sbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 0, nullptr, GL_DYNAMIC_COPY);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &max_group_size);

    createTexture(texWidth, texHeight);

    gpu_initialized = true;
}

void ComputeGPU::uploadData(const std::vector<float>& data) {
    if (!g_program) {
        throw std::runtime_error("GPU not initialized. Call init first.");
    }

    size_t data_size = data.size() * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo);

    if (data_size != g_buffer_size) {
        glBufferData(GL_SHADER_STORAGE_BUFFER, data_size, data.data(), GL_DYNAMIC_COPY);
        g_buffer_size = data_size;
    } else {
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data_size, data.data());
    }

    g_data_on_gpu = true;
}

void ComputeGPU::process(std::vector<float>& data) {
    if (!gpu_initialized) throw std::runtime_error("GPU not initialized.");
    uploadData(data);
    processDataGPU_NoTransfer(data.size(), false);
    downloadData(data);
}

void ComputeGPU::downloadData(std::vector<float>& data) {
    if (!g_data_on_gpu) throw std::runtime_error("No data on GPU to download.");

    GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
    glDeleteSync(sync);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sbo);
    float* ptr = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
    if (!ptr) {
        throw std::runtime_error("Failed to map GPU buffer for reading.");
    }

    std::copy(ptr, ptr + data.size(), data.begin());
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    g_data_on_gpu = false;
}

void ComputeGPU::processDataGPU_NoTransfer(size_t data_count, bool wait_for_completion) {
    if (!g_program) throw std::runtime_error("GPU not initialized.");
    if (!g_data_on_gpu) throw std::runtime_error("No data on GPU. Call uploadData first.");

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sbo);
    glUseProgram(g_program);

    size_t total_workgroups_1D = (data_count + 255) / 256;
    size_t max_limit = static_cast<size_t>(max_group_size);
    GLuint groups_x = (GLuint)std::min(total_workgroups_1D, max_limit);
    GLuint groups_y = (GLuint)((total_workgroups_1D + groups_x - 1) / groups_x);

    GLint groups_x_loc = glGetUniformLocation(g_program, "u_GroupsX");
    if (groups_x_loc != -1) glUniform1ui(groups_x_loc, groups_x);

    uint threadsPerPixel = (data_count + texWidth * texHeight - 1) / (texWidth * texHeight);
    glBindImageTexture(1, texColor, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    
    glUniform1ui(glGetUniformLocation(g_program, "texWidth"), texWidth);
    glUniform1ui(glGetUniformLocation(g_program, "texHeight"), texHeight);
    glUniform1ui(glGetUniformLocation(g_program, "threadsPerPixel"), threadsPerPixel);

    glBindTexture(GL_TEXTURE_2D, texColor);
    
    glDispatchCompute(groups_x, groups_y, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    if (wait_for_completion) {
        GLsync sync = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glClientWaitSync(sync, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED);
        glDeleteSync(sync);
    }
}

void ComputeGPU::shutdown() {
    if (!gpu_initialized) return;

    if (sbo) glDeleteBuffers(1, &sbo);
    if (g_program) glDeleteProgram(g_program);
    if (g_window) glfwDestroyWindow(g_window);

    gpu_initialized = false;
    glfwTerminate();
    g_buffer_size = 0;
}

void ComputeGPU::renderTexture() {
    if (!gpu_initialized) return;

    // Make sure context is current
    glfwMakeContextCurrent(g_window);

    // Set viewport to texture size
    glViewport(0, 0, texWidth, texHeight);

    // Clear screen
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Simple textured quad
    static GLuint vao = 0, vbo = 0;
    if (vao == 0) {
        float vertices[] = {
            // positions  // texcoords
            -1.0f, -1.0f, 0.0f, 0.0f,
             1.0f, -1.0f, 1.0f, 0.0f,
             1.0f,  1.0f, 1.0f, 1.0f,
            -1.0f,  1.0f, 0.0f, 1.0f
        };
        GLuint indices[] = {0,1,2, 2,3,0};

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        GLuint ebo;
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0); // position
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1); // texcoord
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    }

    // Simple shader to display texture
    static GLuint shaderProgram = 0;
    if (shaderProgram == 0) {
        const char* vsSource = R"(
            #version 430
            layout(location=0) in vec2 pos;
            layout(location=1) in vec2 tex;
            out vec2 vTex;
            void main() { vTex = tex; gl_Position = vec4(pos, 0.0, 1.0); }
        )";
        const char* fsSource = R"(
            #version 430
            in vec2 vTex;
            out vec4 FragColor;
            uniform sampler2D tex0;
            void main() { FragColor = texture(tex0, vTex); }
        )";

        GLuint vs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vs, 1, &vsSource, nullptr);
        glCompileShader(vs);

        GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fs, 1, &fsSource, nullptr);
        glCompileShader(fs);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vs);
        glAttachShader(shaderProgram, fs);
        glLinkProgram(shaderProgram);
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    glUseProgram(shaderProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texColor);
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0);

    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(g_window);
    glfwPollEvents();
}
