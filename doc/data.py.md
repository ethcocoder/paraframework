# Data Utils - Line by Line Explanation

**File:** `modules/framework/data.py`

This module provides standard data handling abstractions compatible with the PyTorch API style.

## Class: Dataset

```python
8: class Dataset:
```
**Lines 8-16:** Abstract Base Class. Requires implementation of `__getitem__` and `__len__`.

## Class: TensorDataset

```python
18: class TensorDataset(Dataset):
```
**Lines 18-30:** A Dataset that wraps pre-loaded tensors. Useful when all data fits in memory. Checks that all input tensors share the same length (dim 0).

## Class: DataLoader

```python
32: class DataLoader:
...
36:     def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
```
**Lines 32-41:** Initialization. Configures batching and shuffling.

```python
42:     def __iter__(self):
43:         n = len(self.dataset)
...
47:             np.random.shuffle(indices)
```
**Lines 42-83:** Iterator logic.
- **Lines 65-71:** **Optimization**. If the underlying dataset is a simple list or `TensorDataset`, it uses array slicing (`dataset[indices]`) instead of a loop. This is much faster than calling `__getitem__` 32 times for a batch of 32.
- **Lines 72-82:** Fallback. If dataset is complex (e.g., loads images from disk), loops one by one and stacks the results.
