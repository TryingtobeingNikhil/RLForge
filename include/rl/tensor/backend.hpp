#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

namespace rl::tensor {

enum class BackendDevice { CPU, GPU };

// Compute backend used by dense tensor kernels. Tensor storage intentionally
// remains host-resident so switching backends never changes Tensor's public
// value or autograd semantics. Accelerated backends may stage buffers on a
// device internally.
class TensorBackend {
public:
    virtual ~TensorBackend() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual BackendDevice device() const noexcept = 0;

    // Row-major matrix multiplication: C[M,N] = A[M,K] * B[K,N].
    virtual void matmul(const double* a, const double* b, double* c,
                        int64_t m, int64_t k, int64_t n) const = 0;
};

// Process-wide backend selection. Access is synchronized and each operation
// takes a shared_ptr snapshot, so a backend remains alive while a kernel runs.
std::shared_ptr<const TensorBackend> current_backend();
void set_backend(std::shared_ptr<const TensorBackend> backend);

std::shared_ptr<const TensorBackend> make_cpu_backend();
std::shared_ptr<const TensorBackend> make_blas_backend();
std::shared_ptr<const TensorBackend> make_cuda_backend();

bool blas_backend_available() noexcept;
bool cuda_backend_available() noexcept;

} // namespace rl::tensor
