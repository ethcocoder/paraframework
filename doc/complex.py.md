# Complex Tensor - Line by Line Explanation

**File:** `modules/framework/complex.py`

This module extends the framework to support **Complex Numbers** (`z = a + bi`). This is critical for Quantum-Like processing (wave equations, superpositions) and signal processing (FFT).

## Class: ComplexTensor

```python
9: class ComplexTensor(Tensor):
```
Inherits from the base `Tensor`.

```python
14:     def __init__(self, data, ...):
...
16:         if not np.iscomplexobj(data):
...
18:                 # If passing a list of tuples or real, handle conversion?
```
**Lines 14-30:** Initialization. Checks if input is complex; handles standard Tensor init.

### Properties
```python
32:     @property
33:     def real(self): ...
36:     @property
37:     def imag(self): ...
```
**Lines 32-37:** Extracts Real and Imaginary components as separate Tensors (presumably useful for gradient descent on complex planes).

### Math Operations
```python
39:     def conj(self): ...
43:     def abs(self): ...
47:     def angle(self): ...
```
**Lines 39-49:**
- **`conj()`**: Complex Conjugate (a - bi).
- **`abs()`**: Magnitude (sqrt(a^2 + b^2)).
- **`angle()`**: Phase angle/Arg (arctan(b/a)).

## FFT Functions

```python
51: def fft(input: Tensor):
...
59: def ifft(input: Tensor):
```
**Lines 51-66:** Fast Fourier Transform and Inverse FFT wrappers. They use the underlying backend (numpy/cupy) for the heavy lifting.
