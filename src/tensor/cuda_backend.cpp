#include "rl/tensor/backend.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#if defined(RL_HAS_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace rl::tensor {

#if defined(RL_HAS_CUDA)
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

class CudaBackend final : public TensorBackend {
public:
    CudaBackend() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
    ~CudaBackend() override { cublasDestroy(handle_); }

    std::string_view name() const noexcept override { return "cuda"; }
    BackendDevice device() const noexcept override { return BackendDevice::GPU; }

    void matmul(const double* a, const double* b, double* c,
                int64_t m, int64_t k, int64_t n) const override {
        std::lock_guard lock(mutex_);
        if (!a || !b || !c || m <= 0 || k <= 0 || n <= 0) {
            throw std::invalid_argument(
                "CUDA matmul requires non-null buffers and positive dimensions");
        }
        if (m > std::numeric_limits<int>::max() ||
            k > std::numeric_limits<int>::max() ||
            n > std::numeric_limits<int>::max()) {
            throw std::overflow_error("CUDA matmul dimensions exceed cuBLAS int limits");
        }

        double* device_a = nullptr;
        double* device_b = nullptr;
        double* device_c = nullptr;
        auto checked_bytes = [](int64_t rows, int64_t columns) {
            const auto elements = static_cast<uint64_t>(rows) *
                                  static_cast<uint64_t>(columns);
            if (elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
                throw std::overflow_error("CUDA matmul allocation size overflows");
            }
            return static_cast<size_t>(elements) * sizeof(double);
        };
        const size_t a_bytes = checked_bytes(m, k);
        const size_t b_bytes = checked_bytes(k, n);
        const size_t c_bytes = checked_bytes(m, n);

        try {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes),
                       "cudaMalloc(A)");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes),
                       "cudaMalloc(B)");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), c_bytes),
                       "cudaMalloc(C)");
            check_cuda(cudaMemcpy(device_a, a, a_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(A)");
            check_cuda(cudaMemcpy(device_b, b, b_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(B)");

            const double alpha = 1.0;
            const double beta = 0.0;
            // cuBLAS is column-major. Row-major C=A*B is equivalent to the
            // column-major operation C^T=B^T*A^T over the same buffers.
            check_cublas(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                     static_cast<int>(n), static_cast<int>(m),
                                     static_cast<int>(k), &alpha,
                                     device_b, static_cast<int>(n),
                                     device_a, static_cast<int>(k), &beta,
                                     device_c, static_cast<int>(n)),
                         "cublasDgemm");
            check_cuda(cudaMemcpy(c, device_c, c_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(C)");
        } catch (...) {
            if (device_a) cudaFree(device_a);
            if (device_b) cudaFree(device_b);
            if (device_c) cudaFree(device_c);
            throw;
        }
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
    }

private:
    cublasHandle_t handle_{};
    mutable std::mutex mutex_;
};

} // namespace
#endif

std::shared_ptr<const TensorBackend> make_cuda_backend() {
#if defined(RL_HAS_CUDA)
    if (!cuda_backend_available()) {
        throw std::runtime_error(
            "CUDA backend was compiled but no CUDA device is available");
    }
    return std::make_shared<CudaBackend>();
#else
    throw std::runtime_error(
        "CUDA backend is unavailable; configure with -DRL_ENABLE_CUDA=ON");
#endif
}

bool cuda_backend_available() noexcept {
#if defined(RL_HAS_CUDA)
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

} // namespace rl::tensor
