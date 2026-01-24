# numba_ops.py

## Overview
High-performance JIT (Just-In-Time) compiled operations for the Paradox AI Framework using Numba. It provides specialized kernels for bottlenecks like matrix multiplication and optimizer updates.

## Purpose
`numba_ops.py` bypasses the overhead of the Python interpreter and even some NumPy broadcast overhead by compiling critical functions directly to machine code, achieving 5-10x speedups on CPU.

## Key Features
- **Parallel Execution**: Uses `numba.prange` for automatic multi-threading across available CPU cores.
- **Fast Math**: Enables `fastmath=True` for additional CPU-level optimizations (e.g., SIMD).
- **Graceful Fallback**: If Numba is not installed, it provides a "dummy" decorator that allows the code to run as standard Python/NumPy, ensuring compatibility.
- **Caching**: Compiled kernels are cached on disk to speed up subsequent imports.

## Components

### `matmul_numba(A, B)`
A custom matrix multiplication implementation optimized for different tensor dimensions.
- **2D Support**: Standard matrix-matrix multiplication.
- **3D Support**: Batch matrix multiplication (supports shared or batch-specific second matrix).
- **4D Support**: Specialized for Transformer attention heads (Batch, Heads, Seq, Dim).
- **Performance**: Manually unrolls loops and uses parallel threading for the outermost loop.

### `adam_step_numba(params, grads, m, v, lr, beta1, beta2, eps, t)`
A fused Adam update kernel.
- **Operation**: Performs the entire Adam calculation (momentum update, velocity update, bias correction, and parameter subtraction) in a single pass over the data.
- **Efficiency**: Reduces memory bandwidth usage by performing all modifications in-place on flattened arrays.

### `njit` (Dummy Decorator)
Used if Numba is missing. It simply returns the original function, preventing import errors while maintaining the API of the file.

## Usage
These functions are typically not called directly by the user but are used internally by:
- `optim.py` (for `adam_step_numba`)
- `tensor.py` (as a fast-path for `@` / `matmul` operations)

## Diagnostic Utility
- **`test_numba_ops()`**: A built-in test suite that verifies the numerical accuracy of the Numba kernels against standard NumPy results for 2D, 3D, and 4D tensors.

## Performance Comparison
- **Standard NumPy**: Good for large matrices, high overhead for small operations in loops.
- **Numba Ops**: Excellent for mid-sized operations and fused loops (like Adam), often outperforming NumPy by avoiding temporary array allocations.
