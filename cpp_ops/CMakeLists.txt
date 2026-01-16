cmake_minimum_required(VERSION 3.15)
project(paradox_fast_ops)

set(CMAKE_CXX_STANDARD 17)

# MSVC optimizations (Windows)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2")
endif()

# GCC/Clang optimizations (Linux/Mac)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# Create Python module
pybind11_add_module(fast_ops fast_ops.cpp)

# Set output directory to parent (modules/framework/)
set_target_properties(fast_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)
