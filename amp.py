
_AMP_ENABLED = False

class autocast:
    """
    Context manager for Automatic Mixed Precision (AMP).
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.prev = False
        
    def __enter__(self):
        global _AMP_ENABLED
        self.prev = _AMP_ENABLED
        _AMP_ENABLED = self.enabled
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _AMP_ENABLED
        _AMP_ENABLED = self.prev

def is_autocast_enabled():
    return _AMP_ENABLED
