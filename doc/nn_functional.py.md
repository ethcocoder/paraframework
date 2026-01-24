# nn/functional.py

## Overview
Stateless functional versions of neural network operations. It contains mathematical implementations of activations and loss functions that do not maintain internal state/parameters.

## Purpose
`functional.py` provides the logic for complex mathematical routines used in deep learning. These are typically called by `Module` classes or directly in custom training loops.

## Key Functions

### `log_softmax(input, axis=-1)`
Computes the logarithm of the softmax function.
- **Numerically Stable**: Uses the LogSumExp trick ($x - \max(x) - \log(\sum \exp(x - \max(x)))$) to prevent overflow/underflow.
- **Usage**: Standard output processing for classification tasks.

### `nll_loss(input, target, reduction='mean', ignore_index=-100)`
Computes the Negative Log Likelihood (NLL) loss.
- **Input**: `input` must be log-probabilities (result of `log_softmax`).
- **Logic**: Selects the log-probability of the correct class for each sample.
- **Differentiation**: Implemented using a "One-Hot Multiplication" trick to stay within the differentiable bounds of the framework's existing operations without needing a dedicated `Gather` operation.
- **Reduction**: Supports `mean` (average over batch) or none (raw losses per sample).

### `cross_entropy(input, target, reduction='mean', ignore_index=-100)`
A convenience function that combines `log_softmax` and `nll_loss`.
- **Input**: Raw logits (unnormalized scores).
- **Behavior**: More stable and convenient than calling the two components separately.

## Implementation Details
- **LogSumExp Trick**: Subtracting the maximum value before exponentiation is a critical design choice for numerical stability in the `log_softmax` implementation.
- **Masked Loss**: Supports an `ignore_index` (default -100) which allows the model to ignore certain tokens during training (e.g., padding tokens in NLP).

## Usage
Typically imported as `F`:
```python
import modules.framework.nn.functional as F

# Inside a forward pass or loss calculation
loss = F.cross_entropy(logits, labels)
```
