from collections import defaultdict
import math
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

from modules.framework.tensor import Tensor
from modules.framework.device import device
import numpy as np # [PARADMA] Replacing Numpy
import logging

logger = logging.getLogger(__name__)

# Try to import Numba accelerated operations
try:
    from modules.framework.numba_ops import adam_step_numba, NUMBA_AVAILABLE
    if NUMBA_AVAILABLE:
        logger.info("[SPEED] Numba JIT acceleration enabled for optimizers")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("[SPEED] Numba not available for optimizers, using NumPy")

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    
    xp = device.backend
    
    # Filter params with grads
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 0.0
    
    # Calculate Total Norm
    total_norm = 0.0
    if norm_type == 2:
        for p in params_with_grad:
            # We use .data to avoid graph building
            grad = p.grad.data
            grad_norm = xp.linalg.norm(grad)
            total_norm += grad_norm ** 2
        total_norm = float(xp.sqrt(total_norm))
    else:
        # Generic p-norm
        for p in params_with_grad:
            grad = p.grad.data
            grad_norm = xp.linalg.norm(grad, ord=norm_type)
            total_norm += grad_norm ** norm_type
        total_norm = float(total_norm ** (1. / norm_type))
        
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad._data *= clip_coef
            
    return total_norm

class Optimizer:
    def __init__(self, params, defaults):
        """
        Base Optimizer.
        Args:
            params: iterable of parameters or dicts defining parameter groups
            defaults: (dict): default values for optimization options
        """
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        
        # Simple param handling: convert to list if it's not a list of dicts
        # We assume simple list of params for this version mostly
        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("Optimizer got an empty parameter list")
            
        if not isinstance(param_list[0], dict):
            self.param_groups.append({'params': param_list})
        else:
            self.param_groups = param_list
            
        # Apply defaults
        for group in self.param_groups:
            for name, default in defaults.items():
                if name not in group:
                    group[name] = default
                    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Optimized zeroing
                    if hasattr(p.grad._data, 'fill'):
                         p.grad._data.fill(0)
                    else:
                         p.grad._data *= 0

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self1, params, lr=0.01, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self):
        xp = device.backend
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Weight Decay
                if weight_decay != 0:
                    d_p = d_p + weight_decay * p.data
                    
                # Momentum
                if momentum != 0:
                    param_state = self.state[id(p)]
                    if 'momentum_buffer' not in param_state:
                         buf = param_state['momentum_buffer'] = xp.zeros_like(p.data)
                         buf += d_p # Initialize
                         d_p = buf
                    else:
                         buf = param_state['momentum_buffer']
                         # buf = momentum * buf + d_p
                         buf *= momentum
                         buf += d_p
                         d_p = buf
                
                if momentum == 0:
                     # Accelerated C-Path
                     from modules.framework.c_bridge import bridge
                     bridge.sgd_step(p._data, d_p, lr, weight_decay)
                     continue
                
                p._data -= lr * d_p

class Adam(Optimizer):
    """
    Implements AdamW algorithm (Decoupled Weight Decay).
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self):
        xp = device.backend
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[id(p)]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = device.backend.zeros_like(p.data) # m
                    param_state['exp_avg_sq'] = device.backend.zeros_like(p.data) # v
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                
                param_state['step'] += 1
                
                # Weight Decay (AdamW style: decoupled)
                # Perform decay on parameter BEFORE adaptation
                if weight_decay != 0:
                    p._data -= lr * weight_decay * p.data
                
                # Use Numba if available (much faster!)
                if NUMBA_AVAILABLE and isinstance(p.data, device.backend.ndarray):
                    adam_step_numba(
                        p._data, grad, exp_avg, exp_avg_sq,
                        lr, beta1, beta2, eps, param_state['step']
                    )
                    continue
                
                # Decay
                # m = beta1 * m + (1 - beta1) * grad
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad
                
                # v = beta2 * v + (1 - beta2) * grad^2
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad * grad)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                
                denom = (xp.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps
                
                step_size = lr / bias_correction1
                
                p._data -= step_size * (exp_avg / denom)

class Lion(Optimizer):
    """
    Implements Lion optimizer (Google's latest optimizer).
    More memory efficient than Adam (only stores momentum, not second moment).
    Paper: https://arxiv.org/abs/2302.06675
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[id(p)]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['exp_avg'] = device.backend.zeros_like(p.data)
                
                exp_avg = param_state['exp_avg']
                
                # Weight decay (decoupled)
                if weight_decay != 0:
                    p._data -= lr * weight_decay * p.data
                
                # Lion update
                # c = sign(beta1 * m + (1 - beta1) * grad)
                # m = beta2 * m + (1 - beta2) * grad
                update = beta1 * exp_avg + (1 - beta1) * grad
                sign_update = device.backend.sign(update)
                p._data -= lr * sign_update
                
                # Update momentum
                exp_avg *= beta2
                exp_avg += (1 - beta2) * grad

class Lamb(Optimizer):
    """
    Implements Lamb optimizer (Layer-wise Adaptive Moments optimizer for Batch training).
    Particularly good for large batch sizes.
    Paper: https://arxiv.org/abs/1904.00962
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self):
        xp = device.backend
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                param_state = self.state[id(p)]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = xp.zeros_like(p.data)
                    param_state['exp_avg_sq'] = xp.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                param_state['step'] += 1
                
                # Adam step
                exp_avg *= beta1
                exp_avg += (1 - beta1) * grad
                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (grad * grad)
                
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                
                # Corrected moments
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                # Adam ratio
                adam_step = exp_avg_corrected / (xp.sqrt(exp_avg_sq_corrected) + eps)
                
                # Add weight decay to adam step
                if weight_decay != 0:
                    adam_step += weight_decay * p.data
                
                # Layer-wise adaptation
                weight_norm = xp.linalg.norm(p.data)
                adam_norm = xp.linalg.norm(adam_step)
                
                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0
                
                # Apply update with trust ratio
                p._data -= lr * trust_ratio * adam_step
