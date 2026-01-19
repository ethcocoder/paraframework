# AMP (Automatic Mixed Precision) - Line by Line Explanation

**File:** `modules/framework/amp.py`

This module implements a context manager for Automatic Mixed Precision training. It sets a global flag `_AMP_ENABLED` which tensor operations check to determine if they should run in lower precision (float16) for speed.

## Code Structure

```python
2: _AMP_ENABLED = False
```
**Line 2:** Global state flag.

```python
4: class autocast:
...
12:     def __enter__(self):
13:         global _AMP_ENABLED
14:         self.prev = _AMP_ENABLED
15:         _AMP_ENABLED = self.enabled
```
**Lines 4-20:** Context manager. Saves previous state, enables AMP on enter, and restores state on exit. This allows nesting `with autocast():` blocks.
