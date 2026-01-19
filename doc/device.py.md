# Device Manager - Line by Line Explanation

**File:** `modules/framework/device.py`

This module manages hardware acceleration (CPU vs GPU). It attempts to load `cupy` for CUDA support.

## Class: Device (Singleton)

```python
3: class Device:
4:     _instance = None
...
6:     def __new__(cls):
```
**Lines 3-10:** Singleton pattern implementation. Only one Device manager exists.

### Initialization & Backend Detection
```python
12:     def _initialize(self):
...
22:             self._cpu_backend = __import__('numpy')
...
32:         try:
33:             import cupy
34:             self._gpu_backend = cupy
```
**Lines 12-38:** Backend loading.
- Tries to import `numpy` as CPU backend.
- Tries to import `cupy` as GPU backend.

### Properties
```python
39:     @property
40:     def backend(self):
41:         if self._current_device == 'gpu' and self._gpu_backend:
42:             return self._gpu_backend
43:         return self._cpu_backend
```
**Lines 39-43:** Returns the active library module (numpy or cupy) so other modules can call math functions (`xp.dot(...)`) without caring about the device.

### Switching
```python
49:     def set_device(self, device_type):
```
**Lines 49-56:** Switches mode. Throws error if GPU requested but CuPy missing.

### Transfers
```python
57:     def to_cpu(self, array):
...
62:     def to_gpu(self, array):
```
**Lines 57-67:** Moves data between Host and Device memory.
