import ctypes
import os
import numpy as np
import logging

logger = logging.getLogger("CBridge")

class CBridge:
    def __init__(self):
        self.lib = None
        self.load_library()

    def load_library(self):
        try:
            dll_path = os.path.join(os.getcwd(), "paradox_c_core.dll")
            if os.path.exists(dll_path):
                self.lib = ctypes.CDLL(dll_path)
                # Define argtypes
                self.lib.c_matmul.argtypes = [
                    ctypes.POINTER(ctypes.c_float), # A
                    ctypes.POINTER(ctypes.c_float), # B
                    ctypes.POINTER(ctypes.c_float), # C
                    ctypes.c_int, # M
                    ctypes.c_int, # N
                    ctypes.c_int  # K
                ]
                self.lib.c_relu.argtypes = [
                    ctypes.POINTER(ctypes.c_float), # input
                    ctypes.POINTER(ctypes.c_float), # output
                    ctypes.c_int  # size
                ]
                self.lib.c_softmax.argtypes = [
                    ctypes.POINTER(ctypes.c_float), # input
                    ctypes.POINTER(ctypes.c_float), # output
                    ctypes.c_int  # size
                ]
                self.lib.c_sgd_step.argtypes = [
                    ctypes.POINTER(ctypes.c_float), # params
                    ctypes.POINTER(ctypes.c_float), # grads
                    ctypes.c_int, # size
                    ctypes.c_float, # lr
                    ctypes.c_float  # weight_decay
                ]
                logger.info("  [OK] Paradox C Core (PCC) linked successfully.")
            else:
                logger.warning("  [WARN] Paradox C Core (PCC) DLL not found. Using fallback.")
        except Exception as e:
            logger.warning(f"  [WARN] Failed to load Paradox C Core: {e}. Using fallback.")

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.lib is None:
            return A @ B
        
        # Ensure float32 and contiguous
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        
        M, K = A.shape
        _, N = B.shape
        C = np.zeros((M, N), dtype=np.float32)
        
        self.lib.c_matmul(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K
        )
        return C

    def relu(self, A: np.ndarray) -> np.ndarray:
        if self.lib is None:
            return np.maximum(0, A)
        
        A = np.ascontiguousarray(A, dtype=np.float32)
        out = np.zeros_like(A)
        self.lib.c_relu(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            A.size
        )
        return out

    def sgd_step(self, params: np.ndarray, grads: np.ndarray, lr: float, weight_decay: float = 0.0):
        # Only use C Core if inputs are CPU Numpy arrays
        # if input is CuPy, 'ctypes' access will fail or be invalid for CPU execution
        if self.lib is None or not isinstance(params, np.ndarray):
            # Fallback
            params -= lr * (grads + weight_decay * params)
            return

        # Ensure correct types (must be contiguous float32)
        # Note: We assume params and grads are ALREADY float32 for speed. 
        # Checking/converting every step defeats the purpose.
        
        try:
            self.lib.c_sgd_step(
                params.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                grads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                params.size,
                float(lr),
                float(weight_decay)
            )
        except Exception:
             # Fallback if ctypes conversion fails
             params -= lr * (grads + weight_decay * params)

# Global bridge instance
bridge = CBridge()
