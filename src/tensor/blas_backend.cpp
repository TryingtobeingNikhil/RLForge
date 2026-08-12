#include "rl/tensor/backend.hpp"

#include <limits>
#include <stdexcept>

#if defined(RL_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(RL_HAS_CBLAS)
#include <cblas.h>
#endif

namespace rl::tensor {

#if defined(RL_HAS_CBLAS)
namespace {
class BlasBackend final : public TensorBackend {
public:
    std::string_view name() const noexcept override { return "blas"; }
    BackendDevice device() const noexcept override { return BackendDevice::CPU; }

    void matmul(const double* a, const double* b, double* c,
                int64_t m, int64_t k, int64_t n) const override {
        if (!a || !b || !c || m <= 0 || k <= 0 || n <= 0) {
            throw std::invalid_argument(
                "BLAS matmul requires non-null buffers and positive dimensions");
        }
        if (m > std::numeric_limits<int>::max() ||
            k > std::numeric_limits<int>::max() ||
            n > std::numeric_limits<int>::max()) {
            throw std::overflow_error("BLAS matmul dimensions exceed CBLAS int limits");
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    1.0, a, static_cast<int>(k), b, static_cast<int>(n),
                    0.0, c, static_cast<int>(n));
    }
};
} // namespace
#endif

std::shared_ptr<const TensorBackend> make_blas_backend() {
#if defined(RL_HAS_CBLAS)
    return std::make_shared<BlasBackend>();
#else
    throw std::runtime_error(
        "BLAS backend is unavailable; configure with -DRL_ENABLE_BLAS=ON");
#endif
}

bool blas_backend_available() noexcept {
#if defined(RL_HAS_CBLAS)
    return true;
#else
    return false;
#endif
}

} // namespace rl::tensor
