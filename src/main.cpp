#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "cpu_compute.h"
#include "gpu_compute.h"
#include "ICompute.h"

int main() {
    int BenchmarkIterations = 2;

    const size_t N = 1ULL << 29;
    std::vector<float> dataCPU(N, 64.0f);
    std::vector<float> dataGPU = dataCPU;

    std::unique_ptr<ComputeGPU> computeGPU = std::make_unique<ComputeGPU>("shaders/compute_shader.glsl");
    std::unique_ptr<ComputeCPU> computeCPU = std::make_unique<ComputeCPU>();

    computeGPU = std::make_unique<ComputeGPU>("shaders/compute_shader.glsl");
    computeGPU->init("shaders/compute_shader.glsl");
    computeGPU->uploadData(dataGPU);

    computeCPU = std::make_unique<ComputeCPU>();

    for (int i = 0; i < BenchmarkIterations; i++) {
        std::cout << "\nRun #" << (i+1) << std::endl;

        auto startCPU = std::chrono::high_resolution_clock::now();
        computeCPU->process(dataCPU);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuTime = endCPU - startCPU;

        auto startGPU = std::chrono::high_resolution_clock::now();
        computeGPU->processDataGPU_NoTransfer(dataGPU.size(), true);
       
        // computeGPU->process(dataGPU.size()); // uploads, processes, downloads data
        auto endGPU = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> gpuTime = endGPU - startGPU;

        std::cout << "CPU time: " << cpuTime.count() << " ms\n";
        std::cout << "GPU time: " << gpuTime.count() << " ms\n";
    }

    auto startGPU = std::chrono::high_resolution_clock::now();
    computeGPU->downloadData(dataGPU);
    auto endGPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpuTime = endGPU - startGPU;
    std::cout << "\nFinal GPU download time: " << gpuTime.count() << " ms\n";

    for (size_t i = 0; i < 5; ++i) {
        std::cout << "dataGPU[" << i << "] = " << dataGPU[i] << std::endl;
        std::cout << "dataCPU[" << i << "] = " << dataCPU[i] << std::endl;
    }
    return 0;
}