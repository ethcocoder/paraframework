# C++ Acceleration - Build Instructions

## ✅ Prerequisites (Already Done)
- [x] pybind11 installed
- [x] cmake installed

## 📋 Commands to Run

### Step 1: Navigate to C++ directory
```powershell
cd C:\Users\fitsum.DESKTOP-JDUVJ6V\Downloads\ultimate paradox\modules\framework\cpp_ops
```

### Step 2: Create build directory
```powershell
mkdir build
cd build
```

### Step 3: Generate Visual Studio project
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64
```
**What this does:** Creates Visual Studio build files

**If you get an error:** You might need Visual Studio Build Tools. Download from:
https://visualstudio.microsoft.com/downloads/
Then select "Desktop development with C++"

### Step 4: Build the C++ extension
```powershell
cmake --build . --config Release
```
**What this does:** Compiles fast_ops.cpp into fast_ops.pyd (Python module)

### Step 5: Copy to framework directory
```powershell
copy Release\fast_ops.pyd ..\..\
```
**What this does:** Moves the compiled module where Python can import it

### Step 6: Verify it works
```powershell
cd ..\..\..\..
python -c "import modules.framework.fast_ops as fast_ops; print('C++ module loaded successfully!')"
```

## 🚨 Troubleshooting

**If cmake fails with "Visual Studio not found":**
```powershell
# Install Visual Studio Build Tools 2022, then try:
cmake .. -G "NMake Makefiles"
nmake
```

**If pybind11 not found:**
```powershell
pip install pybind11 --upgrade
```

## ✅ After Success
Once `fast_ops.pyd` is created, I'll integrate it into your Python framework!
