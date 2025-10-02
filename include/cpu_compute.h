#pragma once
#include <vector>
#include "ICompute.h"

class ComputeCPU : public ICompute {
public:
    ComputeCPU() = default;
    void process(std::vector<float>& data) override;

    ~ComputeCPU() {
    }
};