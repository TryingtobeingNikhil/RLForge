// tests/test_max_last_dim.cpp — Milestone 7 Part B: max_last_dim op tests
//
// Four tests:
//   1. Forward correctness (known inputs, known outputs)
//   2. Backward correctness via numerical gradient checking
//   3. Tie-breaking rule (first occurrence / lowest column index wins)
//   4. Version-guard: in-place mutation after forward throws on backward

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl/tensor/autograd.hpp"
#include "rl/tensor/tensor.hpp"

using Catch::Approx;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ============================================================================
// Numerical gradient checking utility (mirrors test_linear.cpp convention).
// ============================================================================
namespace {

constexpr double kEps = 1e-5;
constexpr double kTol = 1e-5;

Tensor numerical_gradient(Tensor x, std::function<Tensor(Tensor)> f) {
    Tensor grad(x.shape());
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        const double orig = x[i];
        x[i] = orig + kEps;
        const double fp = f(x).item();
        x[i] = orig - kEps;
        const double fm = f(x).item();
        x[i] = orig;  // restore
        grad[i] = (fp - fm) / (2.0 * kEps);
    }
    return grad;
}

void check_grad(Tensor x, std::function<Tensor(Tensor)> f,
                const std::string& name = "") {
    // Analytical gradient.
    auto xc = Tensor::from_data(x.data(), x.shape());
    xc.requires_grad_(true);
    auto loss = f(xc);
    loss.backward();
    REQUIRE(xc.grad() != nullptr);

    // Numerical gradient.
    auto xn      = Tensor::from_data(x.data(), x.shape());
    auto num_grad = numerical_gradient(xn, f);

    REQUIRE(xc.grad()->numel() == num_grad.numel());
    for (int64_t i = 0; i < xc.grad()->numel(); ++i) {
        const double analytic = (*xc.grad())[i];
        const double numeric  = num_grad[i];
        const double denom    = std::max(std::abs(analytic), std::abs(numeric));
        const double rel_err  = (denom < 1e-8) ? std::abs(analytic - numeric)
                                                : std::abs(analytic - numeric) / denom;
        INFO("Gradient mismatch at index " << i << " (" << name << "): "
             << "analytic=" << analytic << ", numeric=" << numeric);
        REQUIRE(rel_err < kTol);
    }
}

}  // namespace

// ============================================================================
// Test B1: Forward correctness — known inputs, known outputs.
// ============================================================================
TEST_CASE("max_last_dim: forward correctness", "[max_last_dim]") {
    // Input: 3 rows, 4 columns each.
    // Row 0: [1.0, 3.0, 2.0, 0.5]  -> max = 3.0 at col 1
    // Row 1: [4.0, 0.5, 2.5, 1.0]  -> max = 4.0 at col 0
    // Row 2: [-1.0, -3.0, -0.5, -2.0]  -> max = -0.5 at col 2
    auto x = Tensor::from_data(
        {1.0, 3.0, 2.0, 0.5,
         4.0, 0.5, 2.5, 1.0,
        -1.0, -3.0, -0.5, -2.0},
        {3, 4});

    auto y = x.max_last_dim();

    REQUIRE(y.shape() == std::vector<int64_t>{3});
    REQUIRE(y[0] == Approx(3.0));
    REQUIRE(y[1] == Approx(4.0));
    REQUIRE(y[2] == Approx(-0.5));
}

// ============================================================================
// Test B2: Backward correctness via numerical gradient checking.
//
// Uses inputs with distinct per-row maximums so there are no ties to
// confuse the numerical differencing (central differences would give 0.5
// at a tie, while analytical gives 1.0 for first occurrence).
// ============================================================================
TEST_CASE("max_last_dim: backward correctness via numerical gradient check",
          "[max_last_dim]") {
    // Row 0: max at col 1 (3.0). Row 1: max at col 0 (4.0).
    auto x = Tensor::from_data(
        {1.0, 3.0, 2.0,
         4.0, 0.5, 2.5},
        {2, 3});

    // f(x) = mean(x.max_last_dim()) — scalar output for backward().
    auto f = [](Tensor t) { return t.max_last_dim().mean(); };

    check_grad(x, f, "max_last_dim backward");
}

// ============================================================================
// Test B3: Tie-breaking rule — first occurrence (lowest column index) wins.
// ============================================================================
TEST_CASE("max_last_dim: tie-breaking uses first occurrence (lowest column)",
          "[max_last_dim]") {
    // Row 0: tie between cols 0 and 1 (both 2.0) -> first occurrence = col 0.
    // Row 1: tie between cols 1 and 2 (both 3.0) -> first occurrence = col 1.
    auto x = Tensor::from_data(
        {2.0, 2.0, 1.0,
         1.0, 3.0, 3.0},
        {2, 3});

    auto y = x.max_last_dim();
    REQUIRE(y.shape() == std::vector<int64_t>{2});
    REQUIRE(y[0] == Approx(2.0));  // max from col 0 (first occurrence)
    REQUIRE(y[1] == Approx(3.0));  // max from col 1 (first occurrence)

    // Verify via backward: gradient routes to col 0 for row 0, col 1 for row 1.
    auto xc = Tensor::from_data(x.data(), x.shape());
    xc.requires_grad_(true);
    auto loss = xc.max_last_dim().mean();
    loss.backward();
    REQUIRE(xc.grad() != nullptr);

    // Row 0: gradient at col 0 = 1/2 = 0.5, at col 1 = 0.0, at col 2 = 0.0
    REQUIRE((*xc.grad())[0] == Approx(0.5));  // row0, col0 (argmax)
    REQUIRE((*xc.grad())[1] == Approx(0.0));  // row0, col1 (tie loser)
    REQUIRE((*xc.grad())[2] == Approx(0.0));  // row0, col2

    // Row 1: gradient at col 0 = 0.0, at col 1 = 0.5 (argmax), at col 2 = 0.0
    REQUIRE((*xc.grad())[3] == Approx(0.0));  // row1, col0
    REQUIRE((*xc.grad())[4] == Approx(0.5));  // row1, col1 (argmax, first occurrence)
    REQUIRE((*xc.grad())[5] == Approx(0.0));  // row1, col2 (tie loser)
}

// ============================================================================
// Test B4: Version guard — in-place mutation of input after forward throws.
//
// Consistent with Milestone 6 pattern (mul, square, relu, matmul).
// ============================================================================
TEST_CASE("max_last_dim: version guard fires on in-place mutation after forward",
          "[max_last_dim][version_guard]") {
    auto x = Tensor::from_data(
        {1.0, 3.0, 2.0,
         4.0, 0.5, 2.5},
        {2, 3});
    x.requires_grad_(true);

    auto loss = x.max_last_dim().mean();

    // Mutate x in-place — bumps its Storage version.
    {
        auto buf = x.data_mutable();
        buf[0] = 99.0;
    }

    // backward() must detect the stale version and throw.
    REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);

    // Confirm the error message follows the Milestone 6 convention.
    try {
        auto x2 = Tensor::from_data({1.0, 3.0, 2.0, 4.0, 0.5, 2.5}, {2, 3});
        x2.requires_grad_(true);
        auto loss2 = x2.max_last_dim().mean();
        x2.data_mutable()[0] = 99.0;
        loss2.backward();
        FAIL("Expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        REQUIRE(msg.find("max_last_dim") != std::string::npos);
        REQUIRE(msg.find("backward") != std::string::npos);
        REQUIRE(msg.find("mutated") != std::string::npos);
    }
}

// ============================================================================
// Test B5: Shape validation — non-2D input throws std::invalid_argument.
// ============================================================================
TEST_CASE("max_last_dim: non-2D input throws invalid_argument", "[max_last_dim]") {
    auto x1d = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(x1d.max_last_dim(), std::invalid_argument);

    auto x3d = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3, 1});
    REQUIRE_THROWS_AS(x3d.max_last_dim(), std::invalid_argument);

    auto x0d = Tensor::zeros({});
    REQUIRE_THROWS_AS(x0d.max_last_dim(), std::invalid_argument);
}
