
# Append to the end of math_ops.py

class Sqrt(_Function):
    """Square root operation."""
    @staticmethod
    def forward(ctx, x):
        import numpy as np
        result = np.sqrt(x.data)
        if ctx:
            ctx.save_for_backward(result)
        return result
    
    def backward(self, grad_output):
        result, = self.saved_tensors
        import numpy as np
        # d/dx sqrt(x) = 1/(2*sqrt(x))
        grad = grad_output / (2 * result)
        return grad,
    
    @staticmethod
    def apply(x):
        from modules.framework import autograd
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Sqrt(x) if should_track else None
        if ctx:
            ctx.needs_input_grad = (x.requires_grad,)
            ctx.parents = (x,)
        
        result_data = Sqrt.forward(ctx, x)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)
