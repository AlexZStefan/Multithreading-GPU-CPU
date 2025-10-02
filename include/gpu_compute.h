#pragma once
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ICompute.h"
#include <string>

class ComputeGPU : public ICompute {
public:
    ComputeGPU(const char* shaderPath) {
    }

    ~ComputeGPU() {
        shutdown();
    }

    void process(std::vector<float>& data) override;
    void processDataGPU_NoTransfer(size_t data_count, bool wait_for_completion);
    void downloadData(std::vector<float>& data);
    void init(const char* shaderPath);
    void shutdown();
    void uploadData(const std::vector<float>& data);
private:
    GLuint createComputeShaderProgram(const char* shaderPath);
    std::string loadShaderSource(const char* filePath);

    GLuint g_program = 0;
    GLFWwindow* g_window = nullptr;
    GLuint sbo = 0;
    GLint max_group_size;
    size_t g_buffer_size = 0;
    bool g_data_on_gpu = false;
    bool gpu_initialized = false;
};