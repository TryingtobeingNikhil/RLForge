// tests/test_linear.cpp — Milestone 6 Parts B/D/E: Broadcasting add, Linear layer, MSE loss.
//
// Uses numerical gradient checking (same pattern as test_tensor.cpp) to verify
// backward correctness for the new ops.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "rl/tensor/tensor.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/nn/linear.hpp"
#include "rl/nn/losses.hpp"

using Catch::Approx;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ============================================================================
// Numerical gradient checking utility (mirrors test_tensor.cpp convention).
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
        x[i] = orig;
        grad[i] = (fp - fm) / (2.0 * kEps);
    }
    return grad;
}

void check_grad(Tensor x, std::function<Tensor(Tensor)> f, const std::string& name = "") {
    auto xc = Tensor::from_data(x.data(), x.shape());
    xc.requires_grad_(true);
    auto loss = f(xc);
    loss.backward();
    REQUIRE(xc.grad() != nullptr);

    auto xn      = Tensor::from_data(x.data(), x.shape());
    auto num_grad = numerical_gradient(xn, f);

    REQUIRE(xc.grad()->numel() == num_grad.numel());
    for (int64_t i = 0; i < xc.grad()->numel(); ++i) {
        const double analytic = (*xc.grad())[i];
        const double numeric  = num_grad[i];
        const double denom    = std::max(std::abs(analytic), std::abs(numeric));
        const double err      = (denom < 1e-8) ? std::abs(analytic - numeric)
                                               : std::abs(analytic - numeric) / denom;
        INFO("Gradient check failed for " << (name.empty() ? "op" : name)
             << " at index " << i
             << ": analytic=" << analytic << " numeric=" << numeric
             << " rel_err=" << err);
        REQUIRE(err < kTol);
    }
}

} // namespace

// ============================================================================
// Part B — Broadcasting Add
// ============================================================================

TEST_CASE("Broadcasting add [B,N]+[N]: forward correctness", "[tensor][broadcast]") {
    // [2,3] + [3] should broadcast the bias across both rows.
    auto A = Tensor::from_data({1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0}, {2, 3});
    auto b = Tensor::from_data({10.0, 20.0, 30.0}, {3});
    auto C = A.add(b);

    REQUIRE(C.shape() == std::vector<int64_t>{2, 3});
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    REQUIRE(C[0] == Approx(11.0));
    REQUIRE(C[1] == Approx(22.0));
    REQUIRE(C[2] == Approx(33.0));
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    REQUIRE(C[3] == Approx(14.0));
    REQUIRE(C[4] == Approx(25.0));
    REQUIRE(C[5] == Approx(36.0));
}

TEST_CASE("Broadcasting add [B,N]+[N]: exact-match add still works (no regression)",
          "[tensor][broadcast]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto b = Tensor::from_data({4.0, 5.0, 6.0}, {3});
    auto c = a.add(b);
    REQUIRE(c[0] == Approx(5.0));
    REQUIRE(c[1] == Approx(7.0));
    REQUIRE(c[2] == Approx(9.0));
}

TEST_CASE("Broadcasting add: incompatible shapes throw std::invalid_argument",
          "[tensor][broadcast]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    auto b = Tensor::from_data({1.0, 2.0}, {2});   // [N] doesn't match col count
    REQUIRE_THROWS_AS(a.add(b), std::invalid_argument);

    auto c = Tensor::from_data({1.0, 2.0}, {2});
    auto d = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(c.add(d), std::invalid_argument);
}

TEST_CASE("Broadcasting add [B,N]+[N]: backward correctness via numerical gradient check",
          "[tensor][broadcast][numgrad]") {
    // Check gradient w.r.t. the [B,N] operand (grad passes through).
    const std::vector<int64_t> A_shape = {3, 4};
    const std::vector<int64_t> b_shape = {4};

    // Gradient w.r.t. A: fix b, vary A.
    auto b_fixed = Tensor::from_data({0.5, -0.5, 1.0, -1.0}, {4});
    check_grad(
        Tensor::from_data({1,2,3,4, 5,6,7,8, 9,10,11,12}, A_shape),
        [&b_fixed](Tensor A) {
            return A.add(b_fixed).square().mean();
        },
        "broadcast_add grad wrt A");

    // Gradient w.r.t. b: fix A, vary b — this is the sum-over-batch gradient.
    auto A_fixed = Tensor::from_data({1,2,3,4, 5,6,7,8, 9,10,11,12}, A_shape);
    check_grad(
        Tensor::from_data({0.5, -0.5, 1.0, -1.0}, b_shape),
        [&A_fixed](Tensor b) {
            return A_fixed.add(b).square().mean();
        },
        "broadcast_add grad wrt b (sum-over-batch)");
}

// ============================================================================
// Part D — Transpose
// ============================================================================

TEST_CASE("Tensor::transpose: forward shape and values", "[tensor][ops]") {
    // [2,3] transposed → [3,2]
    auto A = Tensor::from_data({1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0}, {2, 3});
    auto At = A.transpose();
    REQUIRE(At.shape() == std::vector<int64_t>{3, 2});
    // Column-major layout of At: [A[0,0], A[1,0], A[0,1], A[1,1], ...]
    REQUIRE(At[0] == Approx(1.0));  // [0,0] = A[0,0]
    REQUIRE(At[1] == Approx(4.0));  // [0,1] = A[1,0]
    REQUIRE(At[2] == Approx(2.0));  // [1,0] = A[0,1]
    REQUIRE(At[3] == Approx(5.0));  // [1,1] = A[1,1]
    REQUIRE(At[4] == Approx(3.0));  // [2,0] = A[0,2]
    REQUIRE(At[5] == Approx(6.0));  // [2,1] = A[1,2]
}

TEST_CASE("Tensor::transpose: backward via numerical gradient check",
          "[tensor][ops][numgrad]") {
    check_grad(
        Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3}),
        [](Tensor A) {
            // Use transpose in a graph: mean(transpose(A)^2)
            return A.transpose().square().mean();
        },
        "transpose backward");
}

TEST_CASE("Tensor::transpose: non-2D throws std::invalid_argument", "[tensor][ops]") {
    auto v = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    REQUIRE_THROWS_AS(v.transpose(), std::invalid_argument);
}

// ============================================================================
// Part D — Linear Layer
// ============================================================================

TEST_CASE("Linear: output shape and basic forward", "[linear]") {
    // Known small case: Linear(2, 3) with batch size 4.
    rl::nn::Linear layer(2, 3);
    auto params = layer.parameters();

    REQUIRE(params.size() == 2);
    // weight shape: [out=3, in=2]
    REQUIRE(params[0]->shape() == std::vector<int64_t>{3, 2});
    // bias shape: [out=3]
    REQUIRE(params[1]->shape() == std::vector<int64_t>{3});
    REQUIRE(params[0]->requires_grad());
    REQUIRE(params[1]->requires_grad());

    auto x   = Tensor::zeros({4, 2});
    auto out = layer.forward(x);
    // y = x @ W^T + b; with x=0, y = b (which is 0)
    REQUIRE(out.shape() == std::vector<int64_t>{4, 3});
    for (int64_t i = 0; i < out.numel(); ++i) {
        REQUIRE(out[i] == Approx(0.0));
    }
}

TEST_CASE("Linear: known-value forward with hand-computed expected output", "[linear]") {
    // Set W and b manually to known values to verify the forward formula.
    // Linear(2, 2): W = [[1,2],[3,4]], b = [5,6]
    // x = [[1,0],[0,1],[1,1]], y[i] = x[i] @ W^T + b
    //   y[0] = [1,0] @ [[1,3],[2,4]] + [5,6] = [1,3] + [5,6] = [6,9]  ... wait
    //   W = [[1,2],[3,4]], W^T = [[1,3],[2,4]]
    //   y[0] = [1*1+0*2, 1*3+0*4] + [5,6] = [1,3] + [5,6] = [6,9]
    //   y[1] = [0*1+1*2, 0*3+1*4] + [5,6] = [2,4] + [5,6] = [7,10]
    //   y[2] = [1*1+1*2, 1*3+1*4] + [5,6] = [3,7] + [5,6] = [8,13]
    rl::nn::Linear layer(2, 2);
    // Override weight and bias in-place using data_mutable().
    auto params = layer.parameters();
    {
        auto& w = params[0]->data_mutable();  // shape [2,2], row-major
        w[0] = 1.0; w[1] = 2.0;
        w[2] = 3.0; w[3] = 4.0;
    }
    {
        auto& b = params[1]->data_mutable();
        b[0] = 5.0; b[1] = 6.0;
    }

    auto x = Tensor::from_data({1.0, 0.0,
                                  0.0, 1.0,
                                  1.0, 1.0}, {3, 2});
    // Run under no_grad for the forward-value check (don't want to build a
    // graph on top of the already-mutated parameters).
    Tensor out({3, 2});
    {
        auto g = no_grad();
        out = layer.forward(x);
    }

    REQUIRE(out.shape() == std::vector<int64_t>{3, 2});
    REQUIRE(out[0] == Approx(6.0));
    REQUIRE(out[1] == Approx(9.0));
    REQUIRE(out[2] == Approx(7.0));
    REQUIRE(out[3] == Approx(10.0));
    REQUIRE(out[4] == Approx(8.0));
    REQUIRE(out[5] == Approx(13.0));
}

TEST_CASE("Linear: backward via numerical gradient check on weight and bias",
          "[linear][numgrad]") {
    // This is the key integration test: exercises broadcasting backward (for
    // bias) and matmul backward (for weight) together.
    //
    // Strategy: fix the layer architecture (in=3, out=2, batch=4), set
    // parameters to known values, compute analytical gradient via backward(),
    // compare to numerical gradient (central difference).

    const int64_t B = 4, IN = 3, OUT = 2;

    // Fixed input (not a parameter — treat as constant for this test).
    auto x_data = std::vector<double>{
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6,
        0.7, 0.8, 0.9,
        1.0, 1.1, 1.2};

    // Initial weight and bias values (small, so gradients are well-scaled).
    auto w_init = std::vector<double>{0.1, -0.2, 0.3,
                                       -0.1, 0.2, -0.3};
    auto b_init = std::vector<double>{0.5, -0.5};

    // ---- Gradient w.r.t. weight ----
    // Fix bias to its init value; perturb weight.
    auto b_fixed = Tensor::from_data(b_init, {OUT});

    check_grad(
        Tensor::from_data(w_init, {OUT, IN}),
        [&x_data, &b_fixed](Tensor W) {
            W.requires_grad_(true);
            auto x   = Tensor::from_data(x_data, {B, IN});
            auto out = x.matmul(W.transpose()).add(b_fixed);
            return out.square().mean();
        },
        "Linear weight backward");

    // ---- Gradient w.r.t. bias ----
    // Fix weight; perturb bias.
    auto w_fixed = Tensor::from_data(w_init, {OUT, IN});

    check_grad(
        Tensor::from_data(b_init, {OUT}),
        [&x_data, &w_fixed](Tensor b) {
            b.requires_grad_(true);
            auto x   = Tensor::from_data(x_data, {B, IN});
            auto out = x.matmul(w_fixed.transpose()).add(b);
            return out.square().mean();
        },
        "Linear bias backward (sum-over-batch)");
}

// ============================================================================
// Part E — MSE Loss
// ============================================================================

TEST_CASE("mse_loss: forward value correctness", "[loss]") {
    // pred = [1,2,3], target = [1,1,1], mse = mean((0,1,2)^2) = mean(0,1,4) = 5/3
    auto pred   = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto target = Tensor::from_data({1.0, 1.0, 1.0}, {3});
    auto loss   = rl::nn::mse_loss(pred, target);
    REQUIRE(loss.numel() == 1);
    REQUIRE(loss.item() == Approx(5.0 / 3.0));
}

TEST_CASE("mse_loss: backward via numerical gradient check", "[loss][numgrad]") {
    auto target = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    check_grad(
        Tensor::from_data({1.5, 2.5, 3.5}, {3}),
        [&target](Tensor pred) {
            return rl::nn::mse_loss(pred, target);
        },
        "mse_loss backward");
}
