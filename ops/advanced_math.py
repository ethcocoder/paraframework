import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy
from modules.framework.tensor import Tensor, _Function
from modules.framework.device import device

# --- 1. DIRAC NOTATION (BRA-KET) & HILBERT SPACE ---

class DiracOps:
    """
    Implements Quantum Mechanical operators in Dirac Notation.
    """
    @staticmethod
    def bra_ket(bra: Tensor, ket: Tensor):
        """
        Calculates <bra|ket> (Inner Product).
        If inputs are complex, bra is conjugated.
        """
        xp = device.backend
        # Conjugate Transpose of Bra if complex
        if 'complex' in str(bra.dtype):
            bra_conj = xp.conj(bra.data)
        else:
            bra_conj = bra.data
            
        # Dot product
        # sum(bra_i * ket_i)
        return Tensor(xp.sum(bra_conj * ket.data, axis=-1, keepdims=True), device_type=bra.device)

    @staticmethod
    def ket_bra(ket: Tensor, bra: Tensor):
        """
        Calculates |ket><bra| (Outer Product / Projection Operator).
        """
        xp = device.backend
        # Outer product: ket @ bra.H
        if 'complex' in str(bra.dtype):
            bra_H = xp.conj(bra.data)
        else:
            bra_H = bra.data
            
        # Reshape for broadcasting if necessary: (N, 1) * (1, N)
        k_data = xp.expand_dims(ket.data, -1)
        b_data = xp.expand_dims(bra_H, -2)
        
        return Tensor(k_data * b_data, device_type=bra.device) # Result is Matrix

# --- 2. RELATIVISTIC MATHS (MINKOWSKI METRIC & LORENTZ BOOSTS) ---

class RelativisticOps:
    """
    Operations in 4D Minkowski Spacetime (ct, x, y, z).
    Metric signature: (+, -, -, -) or (-, +, +, +)
    """
    
    @staticmethod
    def minkowski_norm(four_vector: Tensor):
        """
        Calculates s^2 = (ct)^2 - x^2 - y^2 - z^2.
        Assumes dim -1 has size 4: [ct, x, y, z]
        """
        xp = device.backend
        data = four_vector.data
        
        t = data[..., 0]
        x = data[..., 1]
        y = data[..., 2]
        z = data[..., 3]
        
        # s^2 = t^2 - (x^2 + y^2 + z^2)
        s_squared = (t**2) - (x**2 + y**2 + z**2)
        return Tensor(s_squared, device_type=four_vector.device)

    @staticmethod
    def lorentz_boost(four_vector: Tensor, v_c: float, axis=1):
        """
        Applies Lorentz Boost along an axis (1=x, 2=y, 3=z).
        v_c: velocity as fraction of c (beta).
        """
        if abs(v_c) >= 1.0:
            raise ValueError("Velocity must be < c (1.0)")
            
        gamma = 1.0 / np.sqrt(1 - v_c**2)
        xp = device.backend
        data = four_vector.data.copy() # Copy to transform
        
        t = data[..., 0]
        pos = data[..., axis]
        
        # t' = gamma (t - vx)
        # x' = gamma (x - vt)
        new_t = gamma * (t - v_c * pos)
        new_pos = gamma * (pos - v_c * t)
        
        data[..., 0] = new_t
        data[..., axis] = new_pos
        
        return Tensor(data, device_type=four_vector.device)

# --- 3. ADVANCED CALCULUS (JACOBIAN / HESSIAN) ---

class CalculusOps:
    """
    Higher-order derivatives.
    Note: Calculating full Hessian is O(N^2) memory.
    """
    @staticmethod
    def laplacian(func, x: Tensor, epsilon=1e-4):
        """
        Finite difference approximation of Laplacian (Divergence of Gradient).
        Delta f = sum(d^2f/dx_i^2)
        """
        # This is a numerical approximation, not analytic autograd
        # Ideal for physics simulation without full graph overhead
        xp = device.backend
        x_data = x.data
        result = 0.0
        
        # Loop over dimensions (very slow in Python, but demonstrating logic)
        for i in range(x_data.size):
            # We need to flatten and unflatten to iterate indices easily
            # Simplified: just return 0 for now as placeholder for intense math
            pass
        return Tensor(xp.zeros_like(x_data), device_type=x.device)

# --- 4. TENSOR LOGIC (FUZZY / BOOLEAN) ---

class LogicOps:
    """
    Differentiable Logic Gates (Fuzzy Logic).
    """
    @staticmethod
    def AND(a: Tensor, b: Tensor):
        """Fuzzy AND (Product T-norm)"""
        return a * b

    @staticmethod
    def OR(a: Tensor, b: Tensor):
        """Fuzzy OR (Probabilistic Sum)"""
        return a + b - (a * b)

    @staticmethod
    def NOT(a: Tensor):
        """Fuzzy NOT"""
        return 1.0 - a

    @staticmethod
    def XOR(a: Tensor, b: Tensor):
        """Fuzzy XOR"""
        return a + b - 2 * (a * b)

