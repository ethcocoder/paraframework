
_GRAD_ENABLED = True

class no_grad:
    """
    Context-manager that disabled gradient calculation.
    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call Tensor.backward(). It will reduce memory consumption
    for computations that would otherwise have requires_grad=True.
    """
    def __init__(self):
        self.prev = True
        
    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev

class enable_grad:
    def __init__(self):
        self.prev = False
        
    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = True
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev

def is_grad_enabled():
    return _GRAD_ENABLED
