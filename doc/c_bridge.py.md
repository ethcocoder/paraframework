# C Bridge - Line by Line Explanation

**File:** `modules/framework/c_bridge.py`

This file is the **Foreign Function Interface (FFI)** layer. It connects the Python framework to the optimized C implementation (`paradox_c_core.dll`).

## Class: CBridge

```python
8: class CBridge:
9:     def __init__(self):
...
13:     def load_library(self):
...
15:             dll_path = os.path.join(os.getcwd(), "paradox_c_core.dll")
16:             if os.path.exists(dll_path):
17:                 self.lib = ctypes.CDLL(dll_path)
```
**Lines 13-49:** Loads the DLL and defines the `argtypes` (C functions signatures) using `ctypes`. This ensures Python passes data correctly to C.

### Operations
```python
50:     def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
...
62:         self.lib.c_matmul(...)
```
**Lines 50-68:** `matmul`. If the DLL is loaded, it dispatches to C. Otherwise, it falls back to `A @ B` (standard numpy).

```python
70:     def relu(self, A: np.ndarray) -> np.ndarray:
```
**Lines 70-81:** Optimized ReLU implementation.

```python
83:     def sgd_step(self, params: np.ndarray, grads: np.ndarray, ...):
```
**Lines 83-106:** Offloads the Stochastic Gradient Descent update step to C.

```python
108: bridge = CBridge()
```
**Line 108:** Instantiates a global singleton bridge object.
