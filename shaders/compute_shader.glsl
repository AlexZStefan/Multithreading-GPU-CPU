#version 430

layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Data {
    float data[];
};

uniform uint u_GroupsX; // The X-dimension workgroup count passed from C++

void main() {
    // 1. Calculate the total number of threads in one "X-slice" (one full row of workgroups)
    // total_threads_in_x_slice = Workgroup Count X * Local Size X
    // gl_WorkGroupSize.x is the local_size_x (256)
    uint total_threads_in_x_slice = u_GroupsX * gl_WorkGroupSize.x;
    
    // 2. Calculate the final 1D index (idx_1D)
    // The Y-component jumps by the size of a full X-slice (row-major order).
    // The X-component is the offset within that X-slice.
    uint idx_1D = gl_GlobalInvocationID.x + 
                  (gl_GlobalInvocationID.y * total_threads_in_x_slice);

    // 3. Bounds check and computation
    if (idx_1D < data.length()) {
        float x = data[idx_1D];
        x = sqrt(x) + sin(x) * cos(x) + exp(-x * 0.001);
        data[idx_1D] = x;
    }
}