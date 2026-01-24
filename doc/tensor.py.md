# tensor.py

## Overview
The core tensor engine of the Paradox Framework. It provides a multi-dimensional array implementation with automatic differentiation (autograd) support, supporting calculations on both CPU (NumPy) and GPU (CuPy).

## Purpose
`tensor.py` serves as the fundamental building block for all mathematical operations in the framework. It tracks operations to build a dynamic computation graph for backpropagation.

## Key Components

### `_Function` (Class)
The base class for all autograd operations. 
- **Fields**:
  - `parents`: Tensors that were inputs to the operation.
  - `saved_tensors`: Tensors saved during the forward pass for use in the backward pass.
- **Methods**:
  - `save_for_backward(*tensors)`: Saves inputs for the gradient calculation.
  - `forward(*args, **kwargs)`: Performs the actual computation.
  - `backward(*args, **kwargs)`: Computes gradients with respect to inputs.

### `Tensor` (Class)
The primary data structure for storing numerical data and gradients.
- **Initialization**: `Tensor(data, device_type=None, requires_grad=False, _ctx=None)`
- **Device Support**: Automatically uses the current global device if not specified.
- **Autograd Integration**: If `requires_grad` is True, operations on this tensor create a computation graph.

## Key Methods

### Core Properties
- `data`: Returns the underlying numerical data (NumPy/CuPy array).
- `shape`: Returns the dimensions of the tensor.
- `dtype`: Returns the data type.
- `to(device_type)`: Moves the tensor to a specific device (CPU/GPU).

### Autograd
- `backward(grad_output=None)`: Triggers backpropagation from the current tensor. It builds a topological sort of the graph and calls the backward method of each operation.
- `zero_grad()`: Resets the gradient of the tensor to zero.

### Mathematical Operations
The class overrides standard Python operators:
- **Arithmetic**: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__pow__`
- **Matrix**: `__matmul__` (@ operator)
- **Reductions**: `sum`, `mean`, `max`
- **Unary**: `exp`, `log`, `sqrt`, `relu`
- **Transformations**: `reshape`, `transpose`, `T` (shorthand for transpose)

### Factories (Static/Class Methods)
- `uniform(*shape, a=0.0, b=1.0)`: Creates a tensor with values from a uniform distribution.
- `normal(*shape, mean=0.0, std=1.0)`: Creates a tensor with values from a normal distribution.
- `ones(*shape)`: Creates a tensor filled with ones.
- `zeros(*shape)`: Creates a tensor filled with zeros.
- `eye(size)`: Creates an identity matrix.

## Implementation Details
- **Lazy Computation**: Grad accumulation happens only when `backward()` is called.
- **Backend Agnostic**: Uses `device.backend` to dispatch operations to either NumPy or CuPy, allowing the same code to run on CPU and GPU seamlessly.
- **Slicing**: Supports standard Python/NumPy slicing via `__getitem__`.

## Integration
- Used by `Module` in `module.py` to store parameters.
- Optimized by `C-Bridge` for certain operations.
- Accelerated by `Numba` for complex element-wise operations.
