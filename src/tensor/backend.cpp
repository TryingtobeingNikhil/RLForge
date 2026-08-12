#include "rl/tensor/backend.hpp"

#include <mutex>
#include <stdexcept>
#include <utility>

namespace rl::tensor {
namespace {

class CpuBackend final : public TensorBackend {
public:
    std::string_view name() const noexcept override { return "cpu"; }
    BackendDevice device() const noexcept override { return BackendDevice::CPU; }

    void matmul(const double* a, const double* b, double* c,
                int64_t m, int64_t k, int64_t n) const override {
        if (!a || !b || !c || m <= 0 || k <= 0 || n <= 0) {
            throw std::invalid_argument(
                "CPU matmul requires non-null buffers and positive dimensions");
        }
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                double acc = 0.0;
                for (int64_t inner = 0; inner < k; ++inner) {
                    acc += a[row * k + inner] * b[inner * n + col];
                }
                c[row * n + col] = acc;
            }
        }
    }
};

std::mutex s_backend_mutex;
std::shared_ptr<const TensorBackend> s_backend = std::make_shared<CpuBackend>();

} // namespace

std::shared_ptr<const TensorBackend> current_backend() {
    std::lock_guard lock(s_backend_mutex);
    return s_backend;
}

void set_backend(std::shared_ptr<const TensorBackend> backend) {
    if (!backend) {
        throw std::invalid_argument("set_backend requires a non-null backend");
    }
    std::lock_guard lock(s_backend_mutex);
    s_backend = std::move(backend);
}

std::shared_ptr<const TensorBackend> make_cpu_backend() {
    return std::make_shared<CpuBackend>();
}

} // namespace rl::tensor
