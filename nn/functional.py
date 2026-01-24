from modules.framework.tensor import Tensor
from modules.framework.device import device
from modules.framework.ops.math_ops import Exp, Sum, Log, Max
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy

def log_softmax(input, axis=-1):
    """
    Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these separately
    is slower and numerically unstable. This function uses the LogSumExp trick.
    """
    # x - max(x) - log(sum(exp(x - max(x))))
    
    # 1. Max for stability
    # input.max(axis=axis, keepdims=True)
    m = input.max(axis=axis, keepdims=True)
    
    # 2. Subtract max (broadcast)
    shifted = input - m
    
    # 3. Exp
    e = shifted.exp()
    
    # 4. Sum
    s = e.sum(axis=axis, keepdims=True)
    
    # 5. Log
    l = s.log()
    
    # 6. Final Result: shifted - log(sum)
    # (input - m) - l = input - m - l
    # We want to return input - m - l
    return shifted - l

def nll_loss(input, target, reduction='mean', ignore_index=-100):
    """
    The negative log likelihood loss.
    input: Tensor (N, C) - Log-probabilities
    target: Tensor (N) - Class indices
    """
    xp = device.backend
    
    # Validation
    if input.shape[0] != target.shape[0]:
         raise ValueError(f"Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).")

    if not isinstance(target, Tensor):
        target = Tensor(target, device_type=input.device)

    # We need to gather the values at the target indices.
    # loss = -input[i, target[i]]
    
    # Advanced indexing
    # Create indices for the batch dimension
    batch_indices = xp.arange(input.shape[0])
    
    # Get raw data
    input_data = input.data
    target_data = target.data
    
    # Masking ignore_index
    if ignore_index is not None:
         mask = (target_data != ignore_index)
         batch_indices = batch_indices[mask]
         target_data = target_data[mask]
         
         if len(batch_indices) == 0:
              return Tensor(xp.array(0.0), device_type=input.device, requires_grad=input.requires_grad)
              
         # If masking occurred, we need to gather from input carefully or handle gradient sparsity
         # For simplicity in this framework, we'll index into the full array
    
    # Gather
    # We can use the EmbeddingFunction logic (indexing) but we need it for general tensors.
    # Or proper slicing.
    # Since 'input' is the Tensor we want grads for, we need to use a differentiable operation.
    # input[batch_indices, target_data] is effectively a Gather operation.
    # Since we don't have a 'Gather' Function in operations yet, we must implement one or use a trick?
    
    # TRICK: One-hot multiplication?
    # loss = - sum(input * one_hot(target))
    # This is memory intensive but differentiable with existing specific Ops (Mul, Sum)
    
    classes = input.shape[1]
    # Make one hot
    # This happens in backend (numpy/cupy)
    one_hot = xp.zeros((len(batch_indices), classes), dtype=input.dtype)
    one_hot[xp.arange(len(batch_indices)), target_data.astype(int)] = 1.0
    
    # Convert one_hot to Tensor (constant)
    # Note: we must slice input if we masked
    if ignore_index is not None:
        # Slicing input via indices isn't a Function yet? 
        # Actually EmbeddingFunction creates one, but Slicing usually creates a View.
        # Let's assume input[mask] works if implemented, but we likely didn't implement boolean indexing on Tensor.
        
        # Fallback: Just compute on full batch but zero out loss for ignored?
        # Re-calc full one_hot with 0s for ignored
        full_one_hot = xp.zeros((input.shape[0], classes), dtype=input.dtype)
        # Only set 1s where target != ignore_index
        valid_mask = (target.data != ignore_index)
        full_one_hot[valid_mask, target.data[valid_mask].astype(int)] = 1.0
        
        target_tensor = Tensor(full_one_hot, device_type=input.device, requires_grad=False)
        
        # Calculate loss
        # element-wise mul -> sum over classes -> sum over batch
        selected_logs = input * target_tensor # (N, C)
        total_loss = -selected_logs.sum() 
        
        count = valid_mask.sum()
    else:
        target_tensor = Tensor(one_hot, device_type=input.device, requires_grad=False)
        selected_logs = input * target_tensor
        total_loss = -selected_logs.sum()
        count = input.shape[0]

    if reduction == 'mean':
        if count > 0:
            return total_loss / float(count)
        else:
            return total_loss # 0
            
    return total_loss

def cross_entropy(input, target, reduction='mean', ignore_index=-100):
    """
    Combination of LogSoftmax and NLLLoss.
    """
    return nll_loss(log_softmax(input, axis=1), target, reduction, ignore_index)

def mse_loss(input, target, reduction='mean'):
    """
    Mean Squared Error loss
    """
    if not isinstance(target, Tensor):
        target = Tensor(target, device_type=input.device)
    
    diff = input - target
    squared = diff * diff
    
    if reduction == 'mean':
        return squared.mean()
    elif reduction == 'sum':
        return squared.sum()
    else:
        return squared

def l1_loss(input, target, reduction='mean'):
    """
    L1 (Mean Absolute Error) loss
    """
    if not isinstance(target, Tensor):
        target = Tensor(target, device_type=input.device)
    
    diff = input - target
    absolute = diff.abs()
    
    if reduction == 'mean':
        return absolute.mean()
    elif reduction == 'sum':
        return absolute.sum()
    else:
        return absolute

