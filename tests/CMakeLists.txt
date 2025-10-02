# tests/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

# --- Find Google Test ---
find_package(GTest REQUIRED)

# --- Test executable ---
add_executable(runTests
    test_compute.cpp
)

# --- Include directories ---
target_include_directories(runTests PRIVATE ../include)

# --- Link libraries ---
target_link_libraries(runTests
    PS_lib                 # reuse the core project library
    GTest::GTest
    GTest::Main
    pthread
)

# --- Enable CTest ---
enable_testing()
add_test(NAME runTests COMMAND runTests)
