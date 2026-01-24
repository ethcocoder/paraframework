# scheduler.py

## Overview
Learning rate scheduling utilities for the Paradox Framework. It provides mechanisms to adjust the learning rate during the training process to improve convergence and stability.

## Purpose
`scheduler.py` allows the user to define rules for how the learning rate should change over time (epochs). This is critical for preventing divergence early in training and ensuring fine-grained updates as the model nears an optimum.

## Base Class

### `_LRScheduler`
The parent class for all schedulers.
- **`__init__(optimizer, last_epoch=-1)`**: Attaches to an optimizer and records the initial learning rates (`base_lrs`).
- **`step(epoch=None)`**: Advances the scheduler's internal clock and updates the optimizer's learning rate.
- **`state_dict()` / `load_state_dict()`**: Allows saving the scheduler's state (current epoch, etc.) for resuming training.

## Implementations

### `StepLR`
Decays the learning rate by a fixed factor (`gamma`) every few steps.
- **Parameters**:
  - `step_size`: Number of epochs between decays.
  - `gamma`: Multiplicative factor (e.g., 0.1).

### `CosineAnnealingLR`
Adjusts the learning rate using a cosine curve, starting at the initial LR and decreasing to a minimum (`eta_min`) over `T_max` steps.
- **Math**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$
- **Benefit**: Smoothly decreases the learning rate, which is often superior to abrupt steps for fine-tuning.

## Usage Example

```python
from modules.framework.scheduler import CosineAnnealingLR

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(model, optimizer)
    scheduler.step() # Learning rate smoothly decreases
```

## Implementation Highlights
- **Initial LR Tracking**: Automatically detects and stores the starting learning rate from the optimizer's `param_groups`.
- **Group Support**: Correctly handles optimizers that have multiple parameter groups with different learning rates.
