# nn/loss.py

## Overview
Object-oriented wrappers for loss functions. These classes inherit from `Module`, making them easy to integrate into training containers.

## Purpose
`loss.py` provides standardized interfaces for loss calculation, ensuring that losses behave like any other module (trackable, portable across devices).

## Components

### `_Loss` (Base Class)
The common ancestor for all loss modules.
- **Fields**:
  - `reduction`: Specifies the reduction to apply to the output (`'mean'`, `'sum'`, or `None`).

### `NLLLoss` (Class)
The object-oriented interface for Negative Log Likelihood loss.
- **Parameters**: 
  - `reduction`: How to aggregate batch loss.
  - `ignore_index`: Index to skip during loss calculation.
- **Usage**: Internal call to `F.nll_loss`.

### `CrossEntropyLoss` (Class)
The standard loss for multi-class classification.
- **Mechanism**: Combines `LogSoftmax` and `NLLLoss` in a single step.
- **Usage**: Internal call to `F.cross_entropy`.

## Design Pattern
The separation between `functional.py` (logic) and `loss.py` (objects) follows the standard convention of modern AI frameworks, providing flexibility for researchers to choose between a functional or declarative style.

## Usage Example
```python
from modules.framework.nn.loss import CrossEntropyLoss

criterion = CrossEntropyLoss()
loss = criterion(model_logits, target_labels)
loss.backward()
```
