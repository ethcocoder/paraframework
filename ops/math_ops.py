from modules.framework.tensor import Tensor, _Function
from modules.framework.device import device
from modules.framework import amp, autograd

class Add(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data.shape, b.data.shape)
        return a.data + b.data

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        grad_a, grad_b = None, None

        if self.needs_input_grad[0]:
            _grad_a = grad_output
            if _grad_a.shape != shape_a:
                for i in range(len(_grad_a.shape) - len(shape_a)):
                    _grad_a = device.backend.sum(_grad_a, axis=0)
                for i, dim_a in enumerate(shape_a):
                    if dim_a == 1:
                        _grad_a = device.backend.sum(_grad_a, axis=i, keepdims=True)
                _grad_a = device.backend.reshape(_grad_a, shape_a)
            grad_a = _grad_a

        if self.needs_input_grad[1]:
            _grad_b = grad_output
            if _grad_b.shape != shape_b:
                for i in range(len(_grad_b.shape) - len(shape_b)):
                    _grad_b = device.backend.sum(_grad_b, axis=0)
                for i, dim_b in enumerate(shape_b):
                    if dim_b == 1:
                        _grad_b = device.backend.sum(_grad_b, axis=i, keepdims=True)
                _grad_b = device.backend.reshape(_grad_b, shape_b)
            grad_b = _grad_b

        return grad_a, grad_b

    @staticmethod
    def apply(a, b):
        should_track = autograd.is_grad_enabled() and (a.requires_grad or b.requires_grad)
        ctx = Add(a, b) if should_track else None
        if ctx:
             ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
        
        result_data = Add.forward(ctx, a, b)
        
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class MatMul(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data, b.data)
        
        # AMP Support
        if amp.is_autocast_enabled():
             a_data = a.data
             b_data = b.data
             if a_data.dtype != 'float16': a_data = a_data.astype('float16')
             if b_data.dtype != 'float16': b_data = b_data.astype('float16')
             return device.backend.matmul(a_data, b_data)

        return device.backend.matmul(a.data, b.data)

    def backward(self, grad):
        a_data, b_data = self.saved_tensors
        xp = device.backend
        grad_a, grad_b = None, None
        
        def mat_transpose(x):
            if x.ndim < 2: return x 
            return xp.swapaxes(x, -1, -2)
            
        if self.needs_input_grad[0]:
            bt = mat_transpose(b_data)
            grad_a = xp.matmul(grad, bt)
            if grad_a.shape != a_data.shape:
                while grad_a.ndim > a_data.ndim:
                    grad_a = xp.sum(grad_a, axis=0)
                for i in range(a_data.ndim):
                    if a_data.shape[i] == 1 and grad_a.shape[i] != 1:
                        grad_a = xp.sum(grad_a, axis=i, keepdims=True)
            
        if self.needs_input_grad[1]:
            at = mat_transpose(a_data)
            grad_b = xp.matmul(at, grad)
            if grad_b.shape != b_data.shape:
                while grad_b.ndim > b_data.ndim:
                    grad_b = xp.sum(grad_b, axis=0)
                for i in range(b_data.ndim):
                    if b_data.shape[i] == 1 and grad_b.shape[i] != 1:
                         grad_b = xp.sum(grad_b, axis=i, keepdims=True)

        return grad_a, grad_b

    @staticmethod
    def apply(a, b):
        should_track = autograd.is_grad_enabled() and (a.requires_grad or b.requires_grad)
        ctx = MatMul(a, b) if should_track else None
        if ctx:
            ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
            
        result_data = MatMul.forward(ctx, a, b)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class ReLU(_Function):
    @staticmethod
    def forward(ctx, x):
        if ctx: ctx.save_for_backward(x.data)
        return device.backend.maximum(0, x.data)

    def backward(self, grad_output):
        x_data = self.saved_tensors[0]
        grad_x = grad_output * (x_data > 0).astype(x_data.dtype)
        return grad_x

    @staticmethod
    def apply(x):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = ReLU(x) if should_track else None
        result_data = ReLU.forward(ctx, x)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Sub(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data.shape, b.data.shape)
        return a.data - b.data

    def backward(self, grad_output):
        shape_a, shape_b = self.saved_tensors
        grad_a = grad_output
        grad_b = -grad_output

        if grad_a.shape != shape_a:
            for i in range(len(grad_a.shape) - len(shape_a)):
                grad_a = device.backend.sum(grad_a, axis=0)
            for i, dim_a in enumerate(shape_a):
                if dim_a == 1:
                    grad_a = device.backend.sum(grad_a, axis=i, keepdims=True)
            grad_a = device.backend.reshape(grad_a, shape_a)

        if grad_b.shape != shape_b:
            for i in range(len(grad_b.shape) - len(shape_b)):
                grad_b = device.backend.sum(grad_b, axis=0)
            for i, dim_b in enumerate(shape_b):
                if dim_b == 1:
                    grad_b = device.backend.sum(grad_b, axis=i, keepdims=True)
            grad_b = device.backend.reshape(grad_b, shape_b)

        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        should_track = autograd.is_grad_enabled() and (a.requires_grad or b.requires_grad)
        ctx = Sub(a, b) if should_track else None
        if ctx: ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
            
        result_data = Sub.forward(ctx, a, b)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Mul(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data, b.data)
        return a.data * b.data

    def backward(self, grad_output):
        a_data, b_data = self.saved_tensors
        xp = device.backend
        grad_a, grad_b = None, None

        if self.needs_input_grad[0]:
             grad_a = grad_output * b_data
             if grad_a.shape != a_data.shape:
                 while grad_a.ndim > a_data.ndim: grad_a = xp.sum(grad_a, axis=0)
                 for i in range(a_data.ndim):
                     if a_data.shape[i] == 1 and grad_a.shape[i] != 1:
                         grad_a = xp.sum(grad_a, axis=i, keepdims=True)
                 if grad_a.shape != a_data.shape: grad_a = grad_a.reshape(a_data.shape)
        
        if self.needs_input_grad[1]:
             grad_b = grad_output * a_data
             if grad_b.shape != b_data.shape:
                 while grad_b.ndim > b_data.ndim: grad_b = xp.sum(grad_b, axis=0)
                 for i in range(b_data.ndim):
                     if b_data.shape[i] == 1 and grad_b.shape[i] != 1:
                         grad_b = xp.sum(grad_b, axis=i, keepdims=True)
                 if grad_b.shape != b_data.shape: grad_b = grad_b.reshape(b_data.shape)

        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        should_track = autograd.is_grad_enabled() and (a.requires_grad or b.requires_grad)
        ctx = Mul(a, b) if should_track else None
        if ctx: ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
        result_data = Mul.forward(ctx, a, b)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Div(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data, b.data)
        return a.data / b.data

    def backward(self, grad_output):
        a_data, b_data = self.saved_tensors
        xp = device.backend
        grad_a = grad_output / b_data
        grad_b = grad_output * (-a_data / (b_data ** 2))

        if self.needs_input_grad[0] and grad_a.shape != a_data.shape:
              while grad_a.ndim > a_data.ndim: grad_a = xp.sum(grad_a, axis=0)
              for i in range(a_data.ndim):
                  if a_data.shape[i] == 1 and grad_a.shape[i] != 1:
                      grad_a = xp.sum(grad_a, axis=i, keepdims=True)
              if grad_a.shape != a_data.shape: grad_a = grad_a.reshape(a_data.shape)

        if self.needs_input_grad[1] and grad_b.shape != b_data.shape:
              while grad_b.ndim > b_data.ndim: grad_b = xp.sum(grad_b, axis=0)
              for i in range(b_data.ndim):
                  if b_data.shape[i] == 1 and grad_b.shape[i] != 1:
                      grad_b = xp.sum(grad_b, axis=i, keepdims=True)
              if grad_b.shape != b_data.shape: grad_b = grad_b.reshape(b_data.shape)

        return (grad_a, grad_b)

    @staticmethod
    def apply(a, b):
        should_track = autograd.is_grad_enabled() and (a.requires_grad or b.requires_grad)
        ctx = Div(a, b) if should_track else None
        if ctx: ctx.needs_input_grad = (a.requires_grad, b.requires_grad)
        result_data = Div.forward(ctx, a, b)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Pow(_Function):
    @staticmethod
    def forward(ctx, a, b):
        if ctx: ctx.save_for_backward(a.data, b)
        return device.backend.power(a.data, b)

    def backward(self, grad_output):
        a_data, b = self.saved_tensors
        grad_a = grad_output * b * device.backend.power(a_data, b - 1)
        return (grad_a,)

    @staticmethod
    def apply(a, b):
        is_b_tensor = isinstance(b, Tensor)
        should_track = autograd.is_grad_enabled() and (a.requires_grad or (is_b_tensor and b.requires_grad))
        
        ctx = None
        if should_track:
             if is_b_tensor: ctx = Pow(a, b)
             else: ctx = Pow(a)

        if is_b_tensor: result_data = Pow.forward(ctx, a, b.data)
        else: result_data = Pow.forward(ctx, a, b)
            
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Exp(_Function):
    @staticmethod
    def forward(ctx, x):
        result = device.backend.exp(x.data)
        if ctx: ctx.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result = self.saved_tensors[0]
        grad_x = grad_output * result
        return (grad_x,)

    @staticmethod
    def apply(x):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Exp(x) if should_track else None
        result_data = Exp.forward(ctx, x)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Log(_Function):
    @staticmethod
    def forward(ctx, x):
        if ctx: ctx.save_for_backward(x.data)
        return device.backend.log(x.data)

    def backward(self, grad_output):
        x_data = self.saved_tensors[0]
        grad_x = grad_output / x_data
        return (grad_x,)

    @staticmethod
    def apply(x):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Log(x) if should_track else None
        result_data = Log.forward(ctx, x)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Sum(_Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        if ctx: ctx.save_for_backward(x.data.shape, axis, keepdims)
        return device.backend.sum(x.data, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        x_shape, axis, keepdims = self.saved_tensors
        if axis is None:
            grad_x = device.backend.full(x_shape, grad_output)
        else:
            if not keepdims:
                grad_output = device.backend.expand_dims(grad_output, axis=axis)
            grad_x = device.backend.broadcast_to(grad_output, x_shape)
        return (grad_x,)

    @staticmethod
    def apply(x, axis=None, keepdims=False):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Sum(x) if should_track else None
        result_data = Sum.forward(ctx, x, axis, keepdims)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Reshape(_Function):
    @staticmethod
    def forward(ctx, x, shape):
        if ctx: ctx.save_for_backward(x.data.shape)
        return device.backend.reshape(x.data, shape)

    def backward(self, grad_output):
        original_shape = self.saved_tensors[0]
        grad_x = device.backend.reshape(grad_output, original_shape)
        return (grad_x,)

    @staticmethod
    def apply(x, shape):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Reshape(x) if should_track else None
        result_data = Reshape.forward(ctx, x, shape)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

class Transpose(_Function):
    @staticmethod
    def forward(ctx, x, axes=None):
        if ctx: ctx.save_for_backward(x.data.shape, axes)
        return device.backend.transpose(x.data, axes=axes)

    def backward(self, grad):
        original_shape, axes = self.saved_tensors
        if axes is None:
            grad_x = device.backend.transpose(grad)
        else:
            inverse_axes = tuple(device.backend.argsort(axes))
            grad_x = device.backend.transpose(grad, axes=inverse_axes)
        return Tensor(grad_x)

    @staticmethod
    def apply(x, axes=None):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Transpose(x) if should_track else None
        if ctx: ctx.needs_input_grad = (x.requires_grad,)
        return Tensor(device.backend.transpose(x.data, axes=axes), requires_grad=should_track, _ctx=ctx)

class Max(_Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        y = device.backend.max(x.data, axis=axis, keepdims=keepdims)
        if ctx: ctx.save_for_backward(x.data, y, axis, keepdims)
        return y

    def backward(self, grad_output):
        x_data, y, axis, keepdims = self.saved_tensors
        xp = device.backend
        
        if not keepdims:
            if axis is None:
                y_expanded = y
                grad_expanded = grad_output
            else:
                if isinstance(axis, int):
                    y_expanded = xp.expand_dims(y, axis=axis)
                    grad_expanded = xp.expand_dims(grad_output, axis=axis)
                else: 
                     y_expanded = y
                     grad_expanded = grad_output
        else:
            y_expanded = y
            grad_expanded = grad_output
            
        mask = (x_data == y_expanded)
        grad_input = grad_expanded * mask.astype(x_data.dtype)
        return (grad_input,)

    @staticmethod
    def apply(x, axis=None, keepdims=False):
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Max(x) if should_track else None
        result_data = Max.forward(ctx, x, axis=axis, keepdims=keepdims)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)

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

class Neg(_Function):
    """Unary negation operation."""
    @staticmethod
    def forward(ctx, x):
        return -x.data

    def backward(self, grad_output):
        return -grad_output

    @staticmethod
    def apply(x):
        from modules.framework import autograd
        should_track = autograd.is_grad_enabled() and x.requires_grad
        ctx = Neg(x) if should_track else None
        if ctx:
            ctx.needs_input_grad = (x.requires_grad,)
            ctx.parents = (x,)
        
        result_data = Neg.forward(ctx, x)
        return Tensor(result_data, requires_grad=should_track, _ctx=ctx)
