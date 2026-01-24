# maths.py

## Overview
A high-level interface that exposes advanced mathematical operations from different sub-systems (Dirac, Relativistic, and Fuzzy Logic) under user-friendly names.

## Purpose
`maths.py` acts as a facade, collecting complex operations from `advanced_math.py` and `complex.py` to make them easily accessible in the main framework without requiring deep knowledge of the underlying module structure.

## Exposed Functions

### Dirac / Quantum Operations
- **`bra_ket(bra, ket)`**: Computes the inner product `⟨bra|ket⟩`.
- **`ket_bra(ket, bra)`**: Computes the outer product `|ket⟩⟨bra|`.

### Relativistic Operations
- **`minkowski_distance(v)`**: Computes the norm of a vector in Minkowski space-time.
- **`lorentz_boost(v, speed)`**: Applies a relativistic Lorentz transformation to a vector.

### Fuzzy Logic Operations
- **`fuzzy_and(a, b)`**: Soft AND operation (typically `min(a, b)` or `a * b`).
- **`fuzzy_or(a, b)`**: Soft OR operation (typically `max(a, b)` or `a + b - a*b`).
- **`fuzzy_not(a)`**: Soft NOT operation (`1 - a`).

### Complex Math
- **`ComplexTensor`**: Class for handling numbers with imaginary components.
- **`fft`**: Fast Fourier Transform for complex data.

## Usage
Instead of importing from nested subdirectories, developers can import everything through this central hub:

```python
from modules.framework.maths import bra_ket, lorentz_boost, fuzzy_and

# Usage
result = fuzzy_and(val1, val2)
energy = minkowski_distance(state_vector)
```

## Module Structure Integration
This file aggregates operations from:
- `modules.framework.ops.advanced_math`
- `modules.framework.complex`
