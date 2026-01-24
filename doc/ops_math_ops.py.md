# ops/math_ops.py

## Overview
The heart of the Paradox Framework's autograd system. It contains the implementations of the basic mathematical operations (addition, multiplication, etc.) and their corresponding backward passes for gradient calculation.

## Purpose
Every mathematical operation performed on a `Tensor` is backed by an `_Function` implementation in this module. These classes define how the values are computed in the forward pass and how gradients are propagated for the backward pass.

## Core Operations

### Binary Operations
- **`Add`**: Implements $a + b$. Supports broadcasting.
- **`Sub`**: Implements $a - b$. Supports broadcasting.
- **`Mul`**: Implements element-wise multiplication $a * b$. Supports broadcasting.
- **`Div`**: Implements element-wise division $a / b$. Supports broadcasting.
- **`Pow`**: Implements $a^b$. Supports broadcasting.
- **`MatMul`**: Implements matrix multiplication $A @ B$. Handles batch dimensions and transposed weight logic for backpropagation.

### Unary Operations
- **`Exp`**: Exponential function $e^x$.
- **`Log`**: Natural logarithm $\ln(x)$.
- **`ReLU`**: Rectified Linear Unit ($\max(0, x)$).
- **`Sqrt`**: Square root $\sqrt{x}$. (Extended in `math_ops_sqrt.py`)

### Reductions & Transformations
- **`Sum`**: Sums elements over specified axes. Handles gradient broadcasting during backpropagation.
- **`Max`**: Finds maximum values. Propagates gradients only to the maximum elements. (Extended in `math_ops_max.py`)
- **`Reshape`**: Changes tensor dimensions without changing data.
- **`Transpose`**: Swaps tensor axes.

## Performance Features
- **Numba Acceleration**: Critical operations (like `MatMul`) automatically use Numba-accelerated kernels if available for significant speed improvements on CPU.
- **AMP Support**: Minimal support for Automatic Mixed Precision (AMP) logic to facilitate half-precision calculations.

## Autograd Mechanics
Each class follows a strict structure:
1. **`forward(ctx, *args)`**: Executes the math using `device.backend`. It saves necessary intermediate values into the `ctx` (context) object.
2. **`backward(ctx, grad_output)`**: Uses the saved values and the incoming `grad_output` to compute gradients for each input variable.
3. **`apply(*args)`**: A static helper that manages the creation of the `_Function` instance and the wrapping of results into a new `Tensor` with the correct context.

## Support for Broadcasting
The gradient calculations for `Add`, `Sub`, `Mul`, and `Div` explicitly check for broadcasted dimensions. If an input was expanded during the forward pass, its gradient is summed across the expanded dimensions to restore the original shape.

## Usage
These operations are usually called via the operator Overloads in the `Tensor` class (e.g., `tensor_a + tensor_b` internally calls `Add.apply(tensor_a, tensor_b)`).
