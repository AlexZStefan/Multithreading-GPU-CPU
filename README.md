# Multithreading GPU-CPU Project

This project demonstrates parallel computation using both CPU multithreading and GPU compute shaders. It is implemented in C++ using OpenGL compute shaders and can run on Linux as well as Windows. 
The project supports large datasets and compares CPU vs GPU performance.

---
## Features

- CPU computation using all available cores (std::thread).
- GPU computation using OpenGL Compute Shaders.
- Unified interface via `ICompute` class.
- Unit tests with Google Test framework.
- Works on Linux and Windows.

---

## Build Instructions (Linux)

1. **Install dependencies:**
sudo apt update
sudo apt install build-essential cmake libglfw3-dev libglew-dev libglm-dev

2. **Clone the repository:**

git clone https://github.com/AlexZStefan/Multithreading-GPU-CPU.git
cd Multithreading-GPU-CPU

3. **Build the project using CMake:**
mkdir build
cd build
cmake ..
make

4. ** Run the program / tests:**
./PS
It will perform computations on both CPU and GPU and print timing results.

./runTests

Tests include:
- CPU computation correctness.
- GPU computation correctness.
- Handling invalid shader paths.
