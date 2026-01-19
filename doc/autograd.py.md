# Autograd - Line by Line Explanation

**File:** `modules/framework/autograd.py`

This module controls the gradient calculation state, similar to `torch.autograd`.

## Global State
```python
2: _GRAD_ENABLED = True
```
**Line 2:** By default, gradient calculation is ON.

## Context Managers

```python
4: class no_grad:
...
17:         _GRAD_ENABLED = False
```
**Lines 4-22:** `no_grad`. Disables gradient tracking. Used during inference to save memory and compute.

```python
23: class enable_grad:
...
30:         _GRAD_ENABLED = True
```
**Lines 23-35:** `enable_grad`. Explicitly enables gradients (e.g., inside a `no_grad` block).
