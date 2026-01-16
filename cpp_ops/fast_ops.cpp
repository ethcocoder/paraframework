#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>

namespace py = pybind11;

// Fast Matrix Multiplication using cache-friendly loop ordering
py::array_t<float> matmul(py::array_t<float> a, py::array_t<float> b) {
    auto buf_a = a.request(), buf_b = b.request();
    
    if (buf_a.ndim != 2 || buf_b.ndim != 2)
        throw std::runtime_error("MatMul requires 2D arrays");
    
    int M = buf_a.shape[0], K = buf_a.shape[1];
    int N = buf_b.shape[1];
    
    if (buf_b.shape[0] != K)
        throw std::runtime_error("MatMul dimension mismatch");
    
    auto result = py::array_t<float>({M, N});
    auto buf_c = result.request();
    
    float* ptr_a = (float*)buf_a.ptr;
    float* ptr_b = (float*)buf_b.ptr;
    float* ptr_c = (float*)buf_c.ptr;
    
    // Initialize output to zero
    for (int i = 0; i < M * N; i++) ptr_c[i] = 0.0f;
    
    // Cache-friendly IKJ loop order
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = ptr_a[i * K + k];
            for (int j = 0; j < N; j++) {
                ptr_c[i * N + j] += a_ik * ptr_b[k * N + j];
            }
        }
    }
    
    return result;
}

// Fast Adam Optimizer Step
void adam_step(
    py::array_t<float> params,
    py::array_t<float> grads,
    py::array_t<float> m,
    py::array_t<float> v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t
) {
    auto buf_p = params.request();
    auto buf_g = grads.request();
    auto buf_m = m.request();
    auto buf_v = v.request();
    
    float* p_ptr = (float*)buf_p.ptr;
    float* g_ptr = (float*)buf_g.ptr;
    float* m_ptr = (float*)buf_m.ptr;
    float* v_ptr = (float*)buf_v.ptr;
    
    int size = buf_p.size;
    
    float bias_correction1 = 1.0f - std::pow(beta1, t);
    float bias_correction2 = 1.0f - std::pow(beta2, t);
    
    for (int i = 0; i < size; i++) {
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g_ptr[i];
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g_ptr[i] * g_ptr[i];
        
        float m_hat = m_ptr[i] / bias_correction1;
        float v_hat = v_ptr[i] / bias_correction2;
        
        p_ptr[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

PYBIND11_MODULE(fast_ops, m) {
    m.doc() = "Paradox AI C++ Accelerated Operations";
    
    m.def("matmul", &matmul, "Fast matrix multiplication with cache-friendly loop order",
          py::arg("a"), py::arg("b"));
    
    m.def("adam_step", &adam_step, "Optimized Adam optimizer parameter update",
          py::arg("params"), py::arg("grads"), py::arg("m"), py::arg("v"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("t"));
}
