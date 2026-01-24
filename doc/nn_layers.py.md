# nn/layers.py

## Overview
High-level neural network layers for the Paradox Framework. These classes inherit from `Module` and encapsulate parameters and operations for common architectural components.

## Purpose
`layers.py` provides the building blocks for creating AI models. Instead of manually managing weights and matrix multiplications, developers can use these prefabricated layers.

## Components

### `Linear` (Class)
A standard fully-connected (dense) layer.
- **Operation**: $y = xW + b$ (internally implements $x A^T + b$ style via `MatMul`).
- **Initialization**: 
  - Uses Xavier/Kaiming-like initialization (`uniform` with bounds based on $1/\sqrt{in\_features}$).
  - Automatically manages `weight` and `bias` as `Parameter` objects.
- **Parameters**:
  - `in_features`: Size of input vector.
  - `out_features`: Size of output vector.
  - `bias`: Boolean, whether to include a learnable bias.

### `PatchEmbedding` (Class)
A specialized layer used in Vision Transformers (ViT).
- **Purpose**: Converts an input image into a sequence of flat patches.
- **Mechanism**:
  - Takes an image of shape `(Batch, Channels, Height, Width)`.
  - Splits it into non-overlapping patches of size `patch_size`.
  - Flattens each patch.
  - Projects the flattened patches into an embedding dimension using an internal `Linear` layer.
- **Output Shape**: `(Batch, NumPatches, EmbedDim)`.
- **Validation**: Ensures image dimensions are divisible by the patch size.

## Usage Example

```python
from modules.framework.nn.layers import Linear

# A simple MLP
layer1 = Linear(784, 256)
layer2 = Linear(256, 10)

x = Tensor.normal(1, 784)
hidden = layer1(x).relu()
output = layer2(hidden)
```

## Implementation Notes
- **Operation Dispatch**: Uses the `apply` method of lower-level operations (like `MatMul` and `Add`) to ensure the computation is registered in the autograd graph.
- **Backend Access**: `PatchEmbedding` directly accesses `device.backend` for efficient reshaping and transposition of image data before wrapping the result back into a `Tensor`.
