#version 430
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float data[];
};

layout(rgba32f, binding = 1) uniform image2D colorImage;

uniform uint u_GroupsX;
uniform uint texWidth;
uniform uint texHeight;
uniform uint threadsPerPixel;

void main() {
    uint total_threads_in_x_slice = u_GroupsX * gl_WorkGroupSize.x;
    uint idx_1D = gl_GlobalInvocationID.x +
                  gl_GlobalInvocationID.y * total_threads_in_x_slice;

    if (idx_1D < data.length()) {
        float x = data[idx_1D];
        x = sqrt(x) + sin(x) * cos(x) + exp(-x * 0.001);

        // Determine which pixel this thread contributes to
        uint pixelIndex = idx_1D / threadsPerPixel;
        uint px = pixelIndex % texWidth;
        uint py = min(pixelIndex / texWidth, texHeight - 1);

        vec4 color = vec4(x / 10.0, sin(x), cos(x), 1.0); // map to RGBA
        imageStore(colorImage, ivec2(px, py), color);

        data[idx_1D] = x;
    }
}
