# optim.py

## Overview
Optimization algorithms for training neural networks in the Paradox Framework. It includes implementations of SGD (Stochastic Gradient Descent) and AdamW, along with gradient clipping utilities.

## Purpose
The `optim.py` module provides the logic for updating model parameters based on computed gradients. It abstracts the update rules (learning rate, momentum, weight decay) from the training loop.

## Key Features
- **Numba Acceleration**: Automatically uses JIT-compiled kernels for Adam updates if Numba is available.
- **C-Bridge Support**: Uses high-performance C implementations for simple SGD steps.
- **AdamW Support**: Implements the decoupled weight decay version of the Adam optimizer.
- **Gradient Clipping**: Prevents exploding gradients via norm-based clipping.

## Components

### `clip_grad_norm_(parameters, max_norm, norm_type=2)`
A utility function to scale gradients so their total norm doesn't exceed `max_norm`.
- **Logic**: Computes the global norm across all parameters and scales them if it exceeds the threshold.

### `Optimizer` (Base Class)
The parent class for all optimizers.
- **`__init__(params, defaults)`**: Initializes parameter groups and default settings.
- **`zero_grad()`**: Efficiently resets the gradients of all optimized parameters to zero.
- **`step()`**: Abstract method to be implemented by specific optimizers.

### `SGD` (Class)
Implements Stochastic Gradient Descent with optional momentum and weight decay.
- **Features**: 
  - Integrated with `c_bridge` for fast CPU updates when momentum is not used.
  - Supports standard momentum buffering.

### `Adam` (Class)
Implements the AdamW algorithm (Decoupled Weight Decay).
- **Features**:
  - **State Tracking**: Maintains moving averages of gradients (`exp_avg`) and squared gradients (`exp_avg_sq`).
  - **AdamW Logic**: Performs weight decay directly on weights before applying the adaptive gradient step.
  - **Performance**: Uses `adam_step_numba` from `numba_ops` if possible, which is significantly faster than raw NumPy/CuPy.

## Usage Example

```python
from modules.framework.optim import Adam

# Define parameters to optimize
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Performance Notes
- **Speed**: The optimizers are designed to be "fast-path". They prefer C or Numba implementations to avoid Python loop overhead.
- **Memory**: Adam maintains two state tensors per parameter, doubling the memory requirements for parameters compared to standard SGD.
