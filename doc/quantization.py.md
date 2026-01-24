# quantization.py

## Overview
A module for post-training quantization, enabling the conversion of high-precision `float32` models into efficient `int8` representations.

## Purpose
Large AI models can be memory-intensive and slow on CPU. `quantization.py` provides a path to reduce model size by 4x and increase execution speed by using integer arithmetic.

## Key Components

### `Quantizer` (Class)
The entry point for model compression.
- **`quantize_dynamic(model)`**: 
  - Iterates through the model's sub-modules.
  - Identifies `Linear` layers (those with a `weight` attribute).
  - Calculates a symmetric quantization scale for each weight: `scale = max(abs(weight)) / 127.0`.
  - Converts weights to `int8`: `w_int8 = round(w_fp32 / scale)`.
  - Replaces the `forward` method of the module with a `QuantizedLinearForward` interceptor.

### `QuantizedLinearForward` (Class)
A functional interceptor that replaces the standard linear layer logic.
- **Mechanism**: When the layer is called, it intercepts the input `x`.
- **Restoration**: It dynamically dequantizes the `int8` weights back to `float32` for the calculation: `w_restored = w_int8 * scale`.
- **Computation**: Performs the matrix multiplication using the restored weights and the original bias.

## Current Implementation Strategy
- **"Fake" Quantization for Compute**: While the weights are stored as `int8` (saving 75% of weight memory), the current implementation converts them back to `float32` just before the GPU/CPU matrix multiplication.
- **True INT8 Storage**: The `Parameter` is effectively replaced by an `int8` array and a `float32` scalar scale.

## Performance Benefits
- **Memory**: Drastic reduction in the RAM/VRAM required to hold the model parameters.
- **Disk Space**: Smaller checkpoint sizes.
- **CPU Speed**: Integer conversion and simplified graph navigation reduce overhead.

## Usage Example
```python
from modules.framework.quantization import Quantizer

# Train your model in FP32
model = MyAIModel()
train_model(model)

# Quantize before deployment
quantized_model = Quantizer.quantize_dynamic(model)

# Now it uses int8 weights internally
output = quantized_model(input_data)
```

## Future Enhancements
- **INT8 Kernels**: Interfacing with C-Bridge or Numba to perform the actual matrix multiplication in `int8` space without dequantizing (True Quantization).
- **Activation Quantization**: Quantizing the inputs (`x`) as well as weights.
- **Quantization-Aware Training (QAT)**: Integrating quantization into the training loop for better accuracy.
