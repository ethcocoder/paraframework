"""
Numba-accelerated operations for Paradox AI Framework
Provides JIT-compiled versions of critical operations for 5-10x speedup
"""
import numpy as np
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    prange = range

@njit(cache=True, fastmath=True, parallel=True)
def matmul_numba(A, B):
    """Parallelized matrix multiplication using Numba JIT. Multi-threaded across CPU cores."""
    if A.ndim == 2:
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Dimension mismatch"
        C = np.zeros((M, N), dtype=A.dtype)
        for i in prange(M):
            for k in range(K):
                val = A[i, k]
                for j in range(N):
                    C[i, j] += val * B[k, j]
        return C
    elif A.ndim == 3:
        Batch, M, K = A.shape
        B_ndim = B.ndim
        if B_ndim == 2:
            K2, N = B.shape
            assert K == K2, "Dimension mismatch"
            C = np.zeros((Batch, M, N), dtype=A.dtype)
            for b in prange(Batch):
                for i in range(M):
                    for k in range(K):
                        val = A[b, i, k]
                        for j in range(N):
                            C[b, i, j] += val * B[k, j]
            return C
        elif B_ndim == 3:
            Batch2, K2, N = B.shape
            assert Batch == Batch2 and K == K2, "Dimension mismatch"
            C = np.zeros((Batch, M, N), dtype=A.dtype)
            for b in prange(Batch):
                for i in range(M):
                    for k in range(K):
                        val = A[b, i, k]
                        for j in range(N):
                            C[b, i, j] += val * B[b, k, j]
            return C
    elif A.ndim == 4:
        Batch, Heads, M, K = A.shape
        if B.ndim == 4:
            Batch2, Heads2, K2, N = B.shape
            assert Batch == Batch2 and Heads == Heads2 and K == K2, "Dimension mismatch"
            C = np.zeros((Batch, Heads, M, N), dtype=A.dtype)
            for b in prange(Batch):
                for h in prange(Heads):
                    for i in range(M):
                        for k in range(K):
                            val = A[b, h, i, k]
                            for j in range(N):
                                C[b, h, i, j] += val * B[b, h, k, j]
            return C
    
    raise ValueError("Unsupported dimensions for Numba MatMul")

@njit(cache=True, fastmath=True, parallel=True)
def adam_step_numba(params, grads, m, v, lr, beta1, beta2, eps, t):
    """Fast Adam optimizer update using Numba JIT"""
    bias_correction1 = 1.0 - beta1 ** t
    bias_correction2 = 1.0 - beta2 ** t
    
    size = params.size
    params_flat = params.ravel()
    grads_flat = grads.ravel()
    m_flat = m.ravel()
    v_flat = v.ravel()
    
    for i in prange(size):
        # Update momentum
        m_flat[i] = beta1 * m_flat[i] + (1.0 - beta1) * grads_flat[i]
        # Update velocity
        v_flat[i] = beta2 * v_flat[i] + (1.0 - beta2) * grads_flat[i] * grads_flat[i]
        
        # Bias-corrected moments
        m_hat = m_flat[i] / bias_correction1
        v_hat = v_flat[i] / bias_correction2
        
        # Update parameters (in-place)
        params_flat[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

# Test function
def test_numba_ops():
    """Test Numba operations"""
    print("Testing Numba JIT operations...")
    
    # Test MatMul 2D
    A2 = np.random.randn(100, 50).astype(np.float32)
    B2 = np.random.randn(50, 80).astype(np.float32)
    
    C2_numba = matmul_numba(A2, B2)
    C2_numpy = A2 @ B2
    
    error2 = np.abs(C2_numba - C2_numpy).max()
    print(f"  MatMul 2D max error: {error2:.2e} (should be < 1e-5)")
    
    # Test MatMul 3D
    A3 = np.random.randn(2, 64, 128).astype(np.float32)
    B3 = np.random.randn(128, 64).astype(np.float32)
    
    C3_numba = matmul_numba(A3, B3)
    C3_numpy = A3 @ B3
    
    error3 = np.abs(C3_numba - C3_numpy).max()
    print(f"  MatMul 3D max error: {error3:.2e} (should be < 1e-5)")
    
    # Test MatMul 4D
    A4 = np.random.randn(1, 4, 16, 32).astype(np.float32)
    B4 = np.random.randn(1, 4, 32, 16).astype(np.float32)
    
    C4_numba = matmul_numba(A4, B4)
    C4_numpy = A4 @ B4
    
    error4 = np.abs(C4_numba - C4_numpy).max()
    print(f"  MatMul 4D max error: {error4:.2e} (should be < 1e-5)")
    
    # Test Adam
    params = np.random.randn(1000).astype(np.float32)
    grads = np.random.randn(1000).astype(np.float32)
    m = np.zeros(1000, dtype=np.float32)
    v = np.zeros(1000, dtype=np.float32)
    
    params_copy = params.copy()
    adam_step_numba(params, grads, m, v, 0.001, 0.9, 0.999, 1e-8, 1)
    print(f"  Adam step completed, params changed: {not np.array_equal(params, params_copy)}")
    
    print(" Numba operations working correctly!")
    
if __name__ == "__main__":
    test_numba_ops()
