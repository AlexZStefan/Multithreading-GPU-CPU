#include "cpu_compute.h"
#include "gpu_compute.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

class GpuTestWithShader : public ::testing::Test {
protected:
    std::unique_ptr<ComputeGPU> compute;
    void SetUp() override {
        compute = std::make_unique<ComputeGPU>("shaders/compute_shader.glsl");
        compute->init("shaders/compute_shader.glsl");
    }

    void TearDown() override {
        compute->shutdown();
    }
};

class CpuTest : public ::testing::Test {
protected:
    std::unique_ptr<ComputeCPU> compute;
    void SetUp() override {
        compute = std::make_unique<ComputeCPU>();
    }
    void TearDown() override {
    }
};

TEST_F(CpuTest, CPUComputeMatchesExpected) {
    const size_t N = 1 << 10; 
    float test_value = 64.0f;

    std::vector<float> data(N, test_value);
    float x = std::sqrt(test_value) + std::sin(test_value) * std::cos(test_value) + std::exp(-test_value * 0.001f);

    float expected(x);
    compute->process(data);

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_FLOAT_EQ(data[i], expected);
}

TEST(GpuTestWithoutShader, InvalidShaderPath) {
    std::unique_ptr<ComputeGPU> compute = std::make_unique<ComputeGPU>("shaders/compute_shader.glsl");
    EXPECT_THROW(compute->init("invalid_path.glsl"), std::runtime_error);
    EXPECT_THROW(compute->init(""), std::runtime_error);
    compute->shutdown();
}

// check that GPU processing matches expected results and 
TEST_F(GpuTestWithShader, LargeDataSet) {
    const size_t N = 1 << 10;   

    std::mt19937 rng(12345);  
    std::uniform_real_distribution<float> dist(10.0f, 100.0f);

    float test_value = 64.0f;

    std::vector<float> data(N, test_value);
    float x = std::sqrt(test_value) + std::sin(test_value) * std::cos(test_value) + std::exp(-test_value * 0.001f);
    float expected(x);
    EXPECT_NO_THROW(compute->process(data));
    
    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_FLOAT_EQ(data[i], expected);
}