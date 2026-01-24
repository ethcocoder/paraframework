# ops/fusion.py

## Overview
A performance optimization module that simulates "Operator Fusion." It provides implementations that combine multiple standard operations into a single execution path to reduce memory overhead and Python dispatcher latency.

## Purpose
In standard neural networks, chaining operations (e.g., Linear layer → Bias → ReLU) results in multiple temporary tensors and multiple calls to the backend. `fusion.py` skips these intermediate steps by performing the entire sequence in a single routine.

## Key Components

### `KernelFusion` (Class)

#### `linear_relu_fused(x, w, b=None)`
Combines a Linear transformation (Matrix Multiplication), bias addition, and a ReLU activation.
- **Logic**: 
  1. Computes $output = xW^T$.
  2. If bias exists, adds it in-place to the output.
  3. Applies ReLU ($\max(0, out)$) in-place.
- **Optimization**: By using in-place operations (`out += b` and `xp.maximum(..., out=out)`), it minimizes the allocation of new memory buffers, which is a major bottleneck on GPUs.

#### `softmax_cross_entropy_fused(logits, targets)`
A placeholder for a specialized high-stability loss calculation (functionally similar to the implementation in `nn/functional.py`).

## Technical Strategy
- **Raw Array Access**: Operates on raw NumPy/CuPy arrays (`.data`) rather than `Tensor` objects to bypass the overhead of the autograd tracker for intermediate calculations.
- **In-place Mutation**: Prioritizes modifying existing memory rather than creating new copies.

## Usage in Framework
The framework's `Quantizer` uses this logic to implement "interceptors." When a model is quantized, its standard forward passes are replaced with these fused kernels to maximize the speedup of the compressed model.

## Future Vision
Plans include expanding the fusion engine to automatically detect and fuse chains of the form `Add -> Mul -> Log`, which are common in Transformer architectures (like Multi-Head Attention).
