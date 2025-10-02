#include "cpu_compute.h"
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>
#include <iostream>

void ComputeCPU::process(std::vector<float>& data) {
    auto worker = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            float &x = data[i];
            // Example computation x = 64 -> f(x) = 9.298
            x = std::sqrt(x) + std::sin(x) * std::cos(x) + std::exp(-x * 0.001f); 
        }
    };

    size_t n = data.size();
    unsigned int num_threads = std::thread::hardware_concurrency(); 
    size_t chunk = n / num_threads;

    std::vector<std::thread> threads;

    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk;
        size_t end = (t == num_threads - 1) ? n : start + chunk;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();
}
