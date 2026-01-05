import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy
from modules.framework.device import device

class KernelFusion:
    """
    Simulates Operator Fusion (e.g., Conv+ReLU, Linear+bias+ReLU)
    to reduce memory access overhead.
    """
    @staticmethod
    def linear_relu_fused(x, w, b=None):
        """
        Computes ReLU(x @ w.T + b) in a single pass ideally.
        In NumPy/CuPy, we chain them, but we avoid creating intermediate Tensors via Framework.
        We stick to raw backend arrays.
        """
        xp = device.backend
        
        # 1. MatMul
        out = xp.matmul(x, w.T)
        
        # 2. Bias Add (In-place if possible)
        if b is not None:
            out += b
            
        # 3. ReLU (In-place via maximum)
        # np.maximum(0, out, out=out) is valid in numpy.
        xp.maximum(0, out, out=out)
        
        return out

    @staticmethod
    def softmax_cross_entropy_fused(logits, targets):
        """
        Fused implementation of Loss calculation for stability and speed.
        """
        # Already part of our functional.py optimization, 
        # but here we emphasize avoiding intermediate allocations.
        pass
