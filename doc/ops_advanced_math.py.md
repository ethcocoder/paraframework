# ops/advanced_math.py

## Overview
Specialized mathematical operators for quantum mechanics, relativity, higher-order calculus, and fuzzy logic.

## Purpose
`advanced_math.py` extends the capabilities of the Paradox Framework beyond standard deep learning, enabling simulations and models that require specialized physics or logic-based math.

## Categories

### 1. Dirac Notation & Hilbert Space
- **`DiracOps.bra_ket(bra, ket)`**: Computes the inner product $\langle bra|ket \rangle$. Supports complex conjugation of the "bra" vector.
- **`DiracOps.ket_bra(ket, bra)`**: Computes the outer product $|ket\rangle\langle bra|$, resulting in a projection matrix.

### 2. Relativistic Math
Operates on 4D Minkowski Spacetime vectors (formatted as $[ct, x, y, z]$).
- **`RelativisticOps.minkowski_norm(v)`**: Calculates the spacetime interval $s^2 = (ct)^2 - (x^2 + y^2 + z^2)$.
- **`RelativisticOps.lorentz_boost(v, v_c, axis)`**: Applies a Lorentz transformation to a 4-vector along the X, Y, or Z axis based on velocity $v/c$.

### 3. Advanced Calculus
- **`CalculusOps.laplacian(func, x)`**: Provides a numerical finite-difference approximation of the Laplacian operator ($\nabla^2 f$).

### 4. Tensor Logic (Fuzzy Logic)
Differentiable versions of logical gates, allowing Boolean logic to be integrated into neural networks.
- **`LogicOps.AND(a, b)`**: $a \cdot b$
- **`LogicOps.OR(a, b)`**: $a + b - (a \cdot b)$
- **`LogicOps.NOT(a)`**: $1.0 - a$
- **`LogicOps.XOR(a, b)`**: $a + b - 2(a \cdot b)$

## Implementation Highlights
- **Backend Agnostic**: Uses `device.backend` to ensure multi-physics calculations run on both CPU and GPU.
- **Differentiability**: Fuzzy logic operations are implemented as standard differentiable tensor math, making them compatible with the framework's autograd system.
- **Complex Support**: Dirac operations check for `complex` dtypes to handle quantum mechanical state vectors correctly.

## Usage Example
```python
from modules.framework.ops.advanced_math import LogicOps, RelativisticOps

# Differentiable logic
prob_a = Tensor([0.8])
prob_b = Tensor([0.5])
logical_result = LogicOps.AND(prob_a, prob_b) # Result is 0.4

# Spacetime boosting
event = Tensor([1.0, 0.5, 0, 0]) # t=1, x=0.5
boosted_event = RelativisticOps.lorentz_boost(event, v_c=0.9)
```
