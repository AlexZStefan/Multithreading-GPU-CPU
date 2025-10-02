#pragma once
#include <vector>

class ICompute {
public:    
    virtual ~ICompute() = default;

    virtual void process(std::vector<float>& data) = 0;
};