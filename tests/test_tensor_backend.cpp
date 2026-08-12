#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string_view>

#include "rl/tensor/backend.hpp"
#include "rl/tensor/tensor.hpp"

using Catch::Approx;

namespace {

class CountingBackend final : public rl::tensor::TensorBackend {
public:
    std::string_view name() const noexcept override { return "counting"; }
    rl::tensor::BackendDevice device() const noexcept override {
        return rl::tensor::BackendDevice::CPU;
    }
    void matmul(const double* a, const double* b, double* c,
                int64_t m, int64_t k, int64_t n) const override {
        ++calls;
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                double sum = 0.0;
                for (int64_t inner = 0; inner < k; ++inner) {
                    sum += a[row * k + inner] * b[inner * n + col];
                }
                c[row * n + col] = sum;
            }
        }
    }
    mutable size_t calls = 0;
};

class BackendRestore {
public:
    BackendRestore() : backend_(rl::tensor::current_backend()) {}
    ~BackendRestore() { rl::tensor::set_backend(backend_); }
private:
    std::shared_ptr<const rl::tensor::TensorBackend> backend_;
};

} // namespace

TEST_CASE("Tensor matmul dispatches through the selected backend", "[backend]") {
    BackendRestore restore;
    auto backend = std::make_shared<CountingBackend>();
    rl::tensor::set_backend(backend);
    auto a = rl::tensor::Tensor::from_data({1.0, 2.0}, {1, 2});
    auto b = rl::tensor::Tensor::from_data({3.0, 4.0}, {2, 1});
    auto result = a.matmul(b);
    REQUIRE(backend->calls == 1);
    REQUIRE(result.item() == Approx(11.0));
}

TEST_CASE("Tensor matmul backward uses the forward backend snapshot", "[backend]") {
    BackendRestore restore;
    auto backend = std::make_shared<CountingBackend>();
    rl::tensor::set_backend(backend);
    auto a = rl::tensor::Tensor::from_data({1.0, 2.0}, {1, 2});
    auto b = rl::tensor::Tensor::from_data({3.0, 4.0}, {2, 1});
    a.requires_grad_(true);
    b.requires_grad_(true);

    a.matmul(b).backward();

    REQUIRE(backend->calls == 3);
    REQUIRE(a.grad() != nullptr);
    REQUIRE(b.grad() != nullptr);
}

TEST_CASE("Portable CPU backend is always available", "[backend]") {
    auto backend = rl::tensor::make_cpu_backend();
    REQUIRE(backend->name() == "cpu");
    REQUIRE(backend->device() == rl::tensor::BackendDevice::CPU);
}

TEST_CASE("Unavailable optional backends fail explicitly", "[backend]") {
    if (!rl::tensor::blas_backend_available()) {
        REQUIRE_THROWS(rl::tensor::make_blas_backend());
    }
    if (!rl::tensor::cuda_backend_available()) {
        REQUIRE_THROWS(rl::tensor::make_cuda_backend());
    }
}

TEST_CASE("Available optional backends produce correct matrix products", "[backend]") {
    BackendRestore restore;
    auto verify = [](std::shared_ptr<const rl::tensor::TensorBackend> backend) {
        rl::tensor::set_backend(std::move(backend));
        auto a = rl::tensor::Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
        auto b = rl::tensor::Tensor::from_data({5.0, 6.0, 7.0, 8.0}, {2, 2});
        auto result = a.matmul(b);
        REQUIRE(result[0] == Approx(19.0));
        REQUIRE(result[1] == Approx(22.0));
        REQUIRE(result[2] == Approx(43.0));
        REQUIRE(result[3] == Approx(50.0));
    };

    if (rl::tensor::blas_backend_available()) {
        verify(rl::tensor::make_blas_backend());
    }
    if (rl::tensor::cuda_backend_available()) {
        verify(rl::tensor::make_cuda_backend());
    }
}
