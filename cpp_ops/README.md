# Paradox AI C++ Accelerated Operations

## Building

### Windows (with Visual Studio)
```powershell
cd modules\framework\cpp_ops
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Linux/Mac
```bash
cd modules/framework/cpp_ops
mkdir build
cd build
cmake ..
make -j4
```

The compiled module (`fast_ops.pyd` on Windows, `fast_ops.so` on Linux) will be placed in `modules/framework/`.

## Requirements
- Python 3.7+
- pybind11: `pip install pybind11`
- CMake: `pip install cmake`
- C++ compiler with C++17 support
