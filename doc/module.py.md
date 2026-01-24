# module.py

## Overview
The implementation of the neural network base classes, `Module` and `Parameter`. It provides the structure for building complex AI architectures in a hierarchical manner.

## Purpose
`module.py` allows for the organization of layers and parameters, facilitating easy training, weight saving/loading, and device management.

## Key Components

### `Parameter` (Class)
A subclass of `Tensor` that is automatically recognized by `Module` as an optimizable weight.
- **Behavior**: By default, `requires_grad` is set to `True`.

### `Module` (Base Class)
The base class for all neural network components (layers, full models, etc.).

#### Core Mechanics
- **Hierarchical Tracking**: Automatically tracks sub-modules and parameters assigned as attributes using `__setattr__`.
- **Training Mode**: Tracks whether the module is in training or evaluation mode via the `training` boolean.

#### Key Methods
- **`forward(*args, **kwargs)`**: The method defining the model's logic. Must be overridden by subclasses.
- **`__call__(*args, **kwargs)`**: Allows calling the module like a function, which internally calls `forward`.
- **`parameters()`**: A generator that yields all parameters in the module and all its sub-modules.
- **`zero_grad()`**: Resets gradients for all parameters to zero.
- **`to(device_type)`**: Recursively moves all parameters and sub-modules to the specified device (CPU/GPU).
- **`train()` / `eval()`**: Sets the mode for the module and all children.
- **`state_dict()`**: Returns a flat dictionary containing all parameter data, with keys reflecting the hierarchy (e.g., `layer1.weight`).
- **`load_state_dict(state_dict)`**: Restores parameter data from a dictionary. It handles unwrapping complex structures and device transitions.
- **`half()` / `float()`**: Casts all floating-point parameters to float16 or float32 for mixed-precision training or memory saving.

## Usage Pattern

```python
from modules.framework.module import Module, Parameter
from modules.framework.tensor import Tensor

class MyLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(Tensor.uniform(in_features, out_features))
        self.bias = Parameter(Tensor.zeros(out_features))
        
    def forward(self, x):
        return x @ self.weight + self.bias
```

## Implementation Highlights
- **Ordered Collections**: Uses `collections.OrderedDict` for `_modules` and `_parameters` to ensure deterministic execution and consistent `state_dict` keys.
- **Deep Loading**: `load_state_dict` includes logic to handle `TensorAxiom` wrappers and CuPy-to-NumPy transitions (via `.get()`).
