from modules.framework.tensor import Tensor
from modules.framework.device import device
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "paradma"))

import numpy as np # [PARADMA] Replacing Numpy

class ComplexTensor(Tensor):
    """
    Experimental Tensor supporting native Complex Numbers (Real + Imaginary).
    Useful for FFTs, Quantum State Vectors, and Euler's Formula operations.
    """
    def __init__(self, data, device_type=None, requires_grad=False):
        # Ensure data is complex type
        if not np.iscomplexobj(data):
            try:
                # If passing a list of tuples or real, handle conversion?
                # For now assume numpy array
                if isinstance(data, (list, tuple)):
                    data = np.array(data)
                
                # Check if user meant to cast
                # mostly we inherit standard behavior but allow complex dtype
                pass 
            except:
                pass
                
        super().__init__(data, device_type, requires_grad)
        
    @property
    def real(self):
        return Tensor(self.data.real, device_type=self.device)
        
    @property
    def imag(self):
        return Tensor(self.data.imag, device_type=self.device)

    def conj(self):
        """Conjugate"""
        return Tensor(device.backend.conj(self.data), device_type=self.device)

    def abs(self):
        """Magnitude"""
        return Tensor(device.backend.abs(self.data), device_type=self.device)
        
    def angle(self):
        """Phase angle"""
        return Tensor(device.backend.angle(self.data), device_type=self.device)

def fft(input: Tensor):
    """
    Fast Fourier Transform.
    """
    xp = device.backend
    out_data = xp.fft.fft(input.data)
    return ComplexTensor(out_data, device_type=input.device)

def ifft(input: Tensor):
    """
    Inverse Fast Fourier Transform.
    """
    xp = device.backend
    out_data = xp.fft.ifft(input.data)
    return ComplexTensor(out_data, device_type=input.device)
