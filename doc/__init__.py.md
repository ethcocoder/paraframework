# Framework Package - Explanation

**File:** `modules/framework/__init__.py`

This is the top-level export file for the custom Deep Learning framework underpinning Paradox. It exposes the core building blocks similar to `torch` or `tensorflow`.

## Exports
- **`Tensor`**: The fundamental data structure.
- **`device`**: Backend management (CPU/GPU/Paradma).
- **`ops`**: Mathematical operations (Add, MatMul, ReLU, etc.).
- **`Module`, `Parameter`**: Base classes for neural network layers.
- **`Optimizer`, `SGD`, `Adam`**: Training algorithms.
