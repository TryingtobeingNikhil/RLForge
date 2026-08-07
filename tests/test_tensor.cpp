// tests/test_tensor.cpp — Milestone 5: Tensor + Core Math + Autograd
//
// Test framework: Catch2 v3.5.4 (same as the rest of the test suite).
// All tests are appended to the single rl_tests binary.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "rl/tensor/autograd.hpp"
#include "rl/tensor/tensor.hpp"

using Catch::Approx;
using rl::tensor::NoGradGuard;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ============================================================================
// Numerical gradient checking utility
//
// Implements central-difference numerical gradient:
//   numerical_grad[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)
//
// Not part of the public library API — test-only helper.
// ============================================================================
namespace {

constexpr double kNumGradEps = 1e-5;
constexpr double kNumGradTol = 1e-5;  // relative tolerance

// Compute f(x) where f maps a Tensor to a scalar Tensor.
// Returns the numerical gradient of f w.r.t. x as a Tensor with the same
// shape as x.
Tensor numerical_gradient(Tensor x, std::function<Tensor(Tensor)> f) {
    Tensor grad(x.shape());
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        const double orig = x[i];

        x[i] = orig + kNumGradEps;
        const double fp = f(x).item();

        x[i] = orig - kNumGradEps;
        const double fm = f(x).item();

        x[i] = orig;  // restore
        grad[i] = (fp - fm) / (2.0 * kNumGradEps);
    }
    return grad;
}

// Run a numerical gradient check for op(x) → scalar loss.
// Computes analytical gradient via backward() and compares against
// central-difference numerical gradient for every element of x.
// Uses relative tolerance where |analytic| > 1e-8, absolute otherwise.
void check_grad(Tensor x, std::function<Tensor(Tensor)> f,
                const std::string& name = "") {
    // Analytical gradient via autograd.
    {
        auto xc = Tensor::from_data(x.data(), x.shape());
        xc.requires_grad_(true);
        auto loss = f(xc);
        loss.backward();
        REQUIRE(xc.grad() != nullptr);

        // Numerical gradient (no_grad so we don't build extra graph).
        auto xn = Tensor::from_data(x.data(), x.shape());
        auto num_grad = numerical_gradient(xn, f);

        REQUIRE(xc.grad()->numel() == num_grad.numel());
        for (int64_t i = 0; i < xc.grad()->numel(); ++i) {
            const double analytic = (*xc.grad())[i];
            const double numeric  = num_grad[i];
            const double denom    = std::max(std::abs(analytic), std::abs(numeric));
            const double rel_err  = (denom < 1e-8) ? std::abs(analytic - numeric)
                                                     : std::abs(analytic - numeric) / denom;
            INFO("Gradient check failed for " << (name.empty() ? "op" : name)
                 << " at index " << i
                 << ": analytic=" << analytic << " numeric=" << numeric
                 << " rel_err=" << rel_err);
            REQUIRE(rel_err < kNumGradTol);
        }
    }
}

} // namespace

// ============================================================================
// Section 1 — Shape, construction, factories
// ============================================================================

TEST_CASE("Tensor: shape and numel for 1-D, 2-D, and scalar", "[tensor]") {
    auto t1 = Tensor::zeros({4});
    REQUIRE(t1.ndim() == 1);
    REQUIRE(t1.numel() == 4);
    REQUIRE(t1.shape() == std::vector<int64_t>{4});

    auto t2 = Tensor::zeros({3, 5});
    REQUIRE(t2.ndim() == 2);
    REQUIRE(t2.numel() == 15);
    REQUIRE(t2.shape() == std::vector<int64_t>{3, 5});

    // 0-D scalar (empty shape)
    auto ts = Tensor::zeros({});
    REQUIRE(ts.ndim() == 0);
    REQUIRE(ts.numel() == 1);
    REQUIRE(ts.item() == Approx(0.0));
}

TEST_CASE("Tensor::ones produces a tensor filled with 1.0", "[tensor]") {
    auto t = Tensor::ones({2, 3});
    for (int64_t i = 0; i < t.numel(); ++i) {
        REQUIRE(t[i] == Approx(1.0));
    }
}

TEST_CASE("Tensor::from_data: shape mismatch throws std::invalid_argument", "[tensor]") {
    REQUIRE_THROWS_AS(Tensor::from_data({1.0, 2.0}, {3}), std::invalid_argument);
    REQUIRE_THROWS_AS(Tensor::from_data({1.0, 2.0, 3.0}, {2, 2}), std::invalid_argument);
    // Correct usage — no throw.
    REQUIRE_NOTHROW(Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2}));
}

TEST_CASE("Tensor::item() throws on multi-element tensor", "[tensor]") {
    auto t = Tensor::zeros({3});
    REQUIRE_THROWS_AS(t.item(), std::invalid_argument);
    auto s = Tensor::zeros({1});
    REQUIRE_NOTHROW(s.item());
}

// ============================================================================
// Section 2 — Forward value correctness for every op
// ============================================================================

TEST_CASE("Tensor::add forward: elementwise addition", "[tensor][ops]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto b = Tensor::from_data({4.0, 5.0, 6.0}, {3});
    auto c = a.add(b);
    REQUIRE(c[0] == Approx(5.0));
    REQUIRE(c[1] == Approx(7.0));
    REQUIRE(c[2] == Approx(9.0));
}

TEST_CASE("Tensor::add shape mismatch throws", "[tensor][ops]") {
    auto a = Tensor::zeros({3});
    auto b = Tensor::zeros({4});
    REQUIRE_THROWS_AS(a.add(b), std::invalid_argument);
}

TEST_CASE("Tensor::sub forward: elementwise subtraction", "[tensor][ops]") {
    auto a = Tensor::from_data({5.0, 3.0, 1.0}, {3});
    auto b = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto c = a.sub(b);
    REQUIRE(c[0] == Approx(4.0));
    REQUIRE(c[1] == Approx(1.0));
    REQUIRE(c[2] == Approx(-2.0));
}

TEST_CASE("Tensor::mul (elementwise) forward", "[tensor][ops]") {
    auto a = Tensor::from_data({2.0, 3.0, 4.0}, {3});
    auto b = Tensor::from_data({5.0, 6.0, 7.0}, {3});
    auto c = a.mul(b);
    REQUIRE(c[0] == Approx(10.0));
    REQUIRE(c[1] == Approx(18.0));
    REQUIRE(c[2] == Approx(28.0));
}

TEST_CASE("Tensor::mul (scalar) forward", "[tensor][ops]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto c = a.mul(3.0);
    REQUIRE(c[0] == Approx(3.0));
    REQUIRE(c[1] == Approx(6.0));
    REQUIRE(c[2] == Approx(9.0));
    // Commutative free function
    auto d = 3.0 * a;
    REQUIRE(d[0] == Approx(3.0));
}

TEST_CASE("Tensor::matmul forward: [2,3] @ [3,2] -> [2,2]", "[tensor][ops]") {
    // [1 2 3]   [7  8]   [1*7+2*9+3*11  1*8+2*10+3*12]   [58  64]
    // [4 5 6] @ [9 10] = [4*7+5*9+6*11  4*8+5*10+6*12] = [139 154]
    //           [11 12]
    auto A = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    auto B = Tensor::from_data({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, {3, 2});
    auto C = A.matmul(B);
    REQUIRE(C.shape() == std::vector<int64_t>{2, 2});
    REQUIRE(C[0] == Approx(58.0));
    REQUIRE(C[1] == Approx(64.0));
    REQUIRE(C[2] == Approx(139.0));
    REQUIRE(C[3] == Approx(154.0));
}

TEST_CASE("Tensor::matmul: dimension errors throw std::invalid_argument", "[tensor][ops]") {
    auto A = Tensor::zeros({2, 3});
    auto B = Tensor::zeros({3, 4});
    REQUIRE_NOTHROW(A.matmul(B));

    // Wrong rank
    REQUIRE_THROWS_AS(Tensor::zeros({3}).matmul(B), std::invalid_argument);
    REQUIRE_THROWS_AS(A.matmul(Tensor::zeros({4})), std::invalid_argument);

    // Inner dimension mismatch
    REQUIRE_THROWS_AS(A.matmul(Tensor::zeros({4, 4})), std::invalid_argument);
}

TEST_CASE("Tensor::relu forward: positive pass-through, negative zero", "[tensor][ops]") {
    auto a = Tensor::from_data({-2.0, 0.0, 3.0, -0.5, 1.0}, {5});
    auto b = a.relu();
    REQUIRE(b[0] == Approx(0.0));
    REQUIRE(b[1] == Approx(0.0));
    REQUIRE(b[2] == Approx(3.0));
    REQUIRE(b[3] == Approx(0.0));
    REQUIRE(b[4] == Approx(1.0));
}

TEST_CASE("Tensor::square forward: elementwise x^2", "[tensor][ops]") {
    auto a = Tensor::from_data({-3.0, 0.0, 2.0, 4.0}, {4});
    auto b = a.square();
    REQUIRE(b[0] == Approx(9.0));
    REQUIRE(b[1] == Approx(0.0));
    REQUIRE(b[2] == Approx(4.0));
    REQUIRE(b[3] == Approx(16.0));
}

TEST_CASE("Tensor::mean forward: mean over all elements", "[tensor][ops]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {4});
    auto m = a.mean();
    REQUIRE(m.numel() == 1);
    REQUIRE(m.shape() == std::vector<int64_t>{});  // scalar shape
    REQUIRE(m.item() == Approx(2.5));

    // 2-D input
    auto b = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    REQUIRE(b.mean().item() == Approx(3.5));
}

TEST_CASE("Tensor::gather forward: selects one value per row", "[tensor][ops]") {
    // [[1, 2, 3],
    //  [4, 5, 6]]   indices = [2, 0]   → [3, 4]
    auto src = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    auto idx = Tensor::from_data({2.0, 0.0}, {2});
    auto out = src.gather(idx);
    REQUIRE(out.shape() == std::vector<int64_t>{2});
    REQUIRE(out[0] == Approx(3.0));
    REQUIRE(out[1] == Approx(4.0));
}

TEST_CASE("Tensor::gather: rank/size errors throw std::invalid_argument", "[tensor][ops]") {
    auto src = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});

    // Non-2-D source
    REQUIRE_THROWS_AS(Tensor::zeros({4}).gather(Tensor::zeros({4})), std::invalid_argument);

    // Non-1-D indices
    REQUIRE_THROWS_AS(src.gather(Tensor::zeros({2, 1})), std::invalid_argument);

    // Index length mismatch
    REQUIRE_THROWS_AS(src.gather(Tensor::zeros({3})), std::invalid_argument);

    // Out-of-bounds index
    auto bad_idx = Tensor::from_data({0.0, 5.0}, {2});
    REQUIRE_THROWS_AS(src.gather(bad_idx), std::invalid_argument);
}

// ============================================================================
// Section 3 — requires_grad propagation
// ============================================================================

TEST_CASE("requires_grad propagates: output requires_grad iff any input does", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0}, {2});
    auto b = Tensor::from_data({3.0, 4.0}, {2});
    a.requires_grad_(true);
    // b does NOT require grad

    // Add: a (rg) + b (no rg) → output rg
    auto c = a.add(b);
    REQUIRE(c.requires_grad());
    // b + b: neither requires grad → output does NOT rg
    auto d = b.add(b);
    REQUIRE_FALSE(d.requires_grad());
    // Both require grad → output rg
    b.requires_grad_(true);
    auto e = a.add(b);
    REQUIRE(e.requires_grad());
}

TEST_CASE("requires_grad: single-input ops propagate requires_grad correctly", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    REQUIRE_FALSE(a.relu().requires_grad());
    REQUIRE_FALSE(a.square().requires_grad());
    REQUIRE_FALSE(a.mean().requires_grad());
    REQUIRE_FALSE(a.mul(2.0).requires_grad());

    a.requires_grad_(true);
    REQUIRE(a.relu().requires_grad());
    REQUIRE(a.square().requires_grad());
    REQUIRE(a.mean().requires_grad());
    REQUIRE(a.mul(2.0).requires_grad());
}

TEST_CASE("no_grad() suppresses graph building regardless of requires_grad", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0}, {2});
    a.requires_grad_(true);

    Tensor c({});
    {
        auto guard = no_grad();
        auto b = a.add(a);
        REQUIRE_FALSE(b.requires_grad());
        REQUIRE(b.node() == nullptr);
        c = b.mean();
        REQUIRE_FALSE(c.requires_grad());
    }
    // Guard out of scope — grad mode restored.
    REQUIRE(rl::tensor::grad_mode_enabled());

    // Now ops outside the guard build graph again.
    auto d = a.add(a);
    REQUIRE(d.requires_grad());
    REQUIRE(d.node() != nullptr);
}

TEST_CASE("detach() shares data, severs graph, sets requires_grad=false", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    a.requires_grad_(true);
    auto b = a.square();  // b has a node

    auto d = b.detach();
    REQUIRE_FALSE(d.requires_grad());
    REQUIRE(d.node() == nullptr);
    // Shared storage: same values
    REQUIRE(d[0] == Approx(1.0));
    REQUIRE(d[1] == Approx(4.0));
    REQUIRE(d[2] == Approx(9.0));
    // Mutating d's underlying data also mutates b (shared storage).
    // This is documented behaviour — detach() is not a copy.
    d[0] = 99.0;
    REQUIRE(b[0] == Approx(99.0));
}

// ============================================================================
// Section 4 — Backward: per-op gradient correctness (analytical)
// ============================================================================

TEST_CASE("Backward: add — both inputs receive upstream grad unchanged", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    auto b = Tensor::from_data({4.0, 5.0, 6.0}, {3});
    a.requires_grad_(true);
    b.requires_grad_(true);

    auto c = a.add(b);
    c.backward(Tensor::ones({3}));

    REQUIRE(a.grad() != nullptr);
    REQUIRE(b.grad() != nullptr);
    for (int64_t i = 0; i < 3; ++i) {
        REQUIRE((*a.grad())[i] == Approx(1.0));
        REQUIRE((*b.grad())[i] == Approx(1.0));
    }
}

TEST_CASE("Backward: sub — lhs gets +1, rhs gets -1", "[tensor][autograd]") {
    auto a = Tensor::from_data({5.0, 6.0}, {2});
    auto b = Tensor::from_data({1.0, 2.0}, {2});
    a.requires_grad_(true);
    b.requires_grad_(true);

    auto c = a.sub(b);
    c.backward(Tensor::ones({2}));

    for (int64_t i = 0; i < 2; ++i) {
        REQUIRE((*a.grad())[i] == Approx(1.0));
        REQUIRE((*b.grad())[i] == Approx(-1.0));
    }
}

TEST_CASE("Backward: mul (elementwise) — cross-multiplication gradients", "[tensor][autograd]") {
    auto a = Tensor::from_data({2.0, 3.0}, {2});
    auto b = Tensor::from_data({4.0, 5.0}, {2});
    a.requires_grad_(true);
    b.requires_grad_(true);

    auto c = a.mul(b);
    c.backward(Tensor::ones({2}));

    // da = b, db = a
    REQUIRE((*a.grad())[0] == Approx(4.0));
    REQUIRE((*a.grad())[1] == Approx(5.0));
    REQUIRE((*b.grad())[0] == Approx(2.0));
    REQUIRE((*b.grad())[1] == Approx(3.0));
}

TEST_CASE("Backward: mul (scalar) — grad scaled by scalar", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    a.requires_grad_(true);
    auto c = a.mul(5.0);
    c.backward(Tensor::ones({3}));
    for (int64_t i = 0; i < 3; ++i) {
        REQUIRE((*a.grad())[i] == Approx(5.0));
    }
}

TEST_CASE("Backward: matmul — correct A^T and B^T multiplications", "[tensor][autograd]") {
    // A=[1,2;3,4], B=[5,6;7,8]
    // C = A@B = [19,22;43,50]
    // dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
    // With dL/dC = ones: dL/dA = [[11,15],[11,15]], dL/dB = [[4,4],[6,6]]
    auto A = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
    auto B = Tensor::from_data({5.0, 6.0, 7.0, 8.0}, {2, 2});
    A.requires_grad_(true);
    B.requires_grad_(true);

    auto C = A.matmul(B);
    C.backward(Tensor::ones({2, 2}));

    // dA = dL/dC @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    REQUIRE((*A.grad())[0] == Approx(11.0));
    REQUIRE((*A.grad())[1] == Approx(15.0));
    REQUIRE((*A.grad())[2] == Approx(11.0));
    REQUIRE((*A.grad())[3] == Approx(15.0));
    // dB = A^T @ dL/dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    REQUIRE((*B.grad())[0] == Approx(4.0));
    REQUIRE((*B.grad())[1] == Approx(4.0));
    REQUIRE((*B.grad())[2] == Approx(6.0));
    REQUIRE((*B.grad())[3] == Approx(6.0));
}

TEST_CASE("Backward: relu — pass-through for positive, zero for negative", "[tensor][autograd]") {
    auto a = Tensor::from_data({-1.0, 2.0, -3.0, 4.0}, {4});
    a.requires_grad_(true);
    auto b = a.relu();
    b.backward(Tensor::ones({4}));

    REQUIRE((*a.grad())[0] == Approx(0.0));
    REQUIRE((*a.grad())[1] == Approx(1.0));
    REQUIRE((*a.grad())[2] == Approx(0.0));
    REQUIRE((*a.grad())[3] == Approx(1.0));
}

TEST_CASE("Backward: square — d(x^2)/dx = 2x", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    a.requires_grad_(true);
    auto b = a.square();
    b.backward(Tensor::ones({3}));

    REQUIRE((*a.grad())[0] == Approx(2.0));
    REQUIRE((*a.grad())[1] == Approx(4.0));
    REQUIRE((*a.grad())[2] == Approx(6.0));
}

TEST_CASE("Backward: mean — uniform 1/N gradient", "[tensor][autograd]") {
    auto a = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {4});
    a.requires_grad_(true);
    auto m = a.mean();
    m.backward();  // scalar: no explicit grad needed

    REQUIRE(a.grad() != nullptr);
    for (int64_t i = 0; i < 4; ++i) {
        REQUIRE((*a.grad())[i] == Approx(0.25));
    }
}

TEST_CASE("Backward: gather — scatter-add gradient", "[tensor][autograd]") {
    // src[2,3]: [[1,2,3],[4,5,6]], indices=[2,0] → out=[3,4]
    // dL/d_out = [1,1]. Backward scatter-add:
    //   grad_src[0,2] += 1  (row 0, col 2)
    //   grad_src[1,0] += 1  (row 1, col 0)
    auto src = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    src.requires_grad_(true);
    auto idx = Tensor::from_data({2.0, 0.0}, {2});

    auto out = src.gather(idx);
    out.backward(Tensor::ones({2}));

    REQUIRE(src.grad() != nullptr);
    // Row 0: col 2 gets grad, cols 0,1 get 0
    REQUIRE((*src.grad())[0] == Approx(0.0));  // [0,0]
    REQUIRE((*src.grad())[1] == Approx(0.0));  // [0,1]
    REQUIRE((*src.grad())[2] == Approx(1.0));  // [0,2]
    // Row 1: col 0 gets grad, cols 1,2 get 0
    REQUIRE((*src.grad())[3] == Approx(1.0));  // [1,0]
    REQUIRE((*src.grad())[4] == Approx(0.0));  // [1,1]
    REQUIRE((*src.grad())[5] == Approx(0.0));  // [1,2]
}

TEST_CASE("Backward: gather with DUPLICATE indices accumulates gradient", "[tensor][autograd]") {
    // CORRECTION #4 from code review: gather with duplicate indices in the
    // index tensor must scatter-ADD into the same cell, not overwrite.
    //
    // src[3,2]: [[1,2],[3,4],[5,6]], indices=[1,1,0]
    // out = [2, 4, 5]
    // dL/d_out = [1, 1, 1]. Backward scatter-add:
    //   grad_src[0,1] += 1  (row 0, col 1)
    //   grad_src[1,1] += 1  (row 1, col 1) — col 1 selected twice ACROSS ROWS
    //   grad_src[2,0] += 1  (row 2, col 0)
    //
    // Note: "duplicate" here means two different rows select the same column.
    // Each row's gradient still goes to its own row, so there's no collision
    // within a row — but to make accumulation truly visible, test a 1-row case:
    //
    // Single-row gather of the same column twice is not possible (each row has
    // one index). The duplicate scenario that *does* accumulate is when the
    // SAME (row, col) combination is selected by multiple gather ops on the
    // same source via a graph, or more directly:
    //
    // We verify accumulation by checking that the two contributions to column 1
    // (rows 0 and 1 both select col 1) produce grad[0,1]=1 and grad[1,1]=1
    // (distinct rows, so both independently += 1 → each cell gets 1, no
    // accumulation within a single cell here).
    //
    // The REAL duplicate scenario: to verify += vs = in the scatter,
    // we use a separate test where we manually confirm that a repeated index
    // in a single call produces the right sum. We do this via: two ops that
    // each call gather and backward, verifying the first backward wrote into
    // the buffer and the second accumulated on top.
    //
    // Additionally, test a case where gather indices tensor selects column 0
    // for BOTH rows, creating overlapping column targets to expose any overwrite.
    auto src = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
    src.requires_grad_(true);
    // Both rows select column 0 → out = [1, 3]
    auto idx = Tensor::from_data({0.0, 0.0}, {2});
    auto out = src.gather(idx);
    out.backward(Tensor::ones({2}));

    // Grad scatter-add:
    //   grad_src[0, 0] += 1  (row 0 col 0)
    //   grad_src[1, 0] += 1  (row 1 col 0)
    //   All other cells stay 0
    REQUIRE((*src.grad())[0] == Approx(1.0));  // [0,0]
    REQUIRE((*src.grad())[1] == Approx(0.0));  // [0,1]
    REQUIRE((*src.grad())[2] == Approx(1.0));  // [1,0]
    REQUIRE((*src.grad())[3] == Approx(0.0));  // [1,1]

    // Now prove accumulation (+=) is happening rather than overwrite:
    // Run backward AGAIN without zero_grad — grads should double.
    src.zero_grad();  // Reset first to isolate the second backward call.
    auto src2 = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
    src2.requires_grad_(true);
    // Use a case where a single backward call must += into the same cell:
    // We make a chain: gather(indices=[0,0]) accumulates into col 0 both rows.
    // Then run backward twice to verify the += semantics of accumulate_grad.
    auto idx2 = Tensor::from_data({0.0, 0.0}, {2});
    auto out2 = src2.gather(idx2);
    out2.backward(Tensor::ones({2}));
    // First backward: col-0 cells each get 1.0
    REQUIRE((*src2.grad())[0] == Approx(1.0));
    REQUIRE((*src2.grad())[2] == Approx(1.0));

    // Re-run the forward + backward from a fresh graph to verify accumulation.
    auto src3 = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {2, 2});
    src3.requires_grad_(true);
    auto out3a = src3.gather(Tensor::from_data({0.0, 0.0}, {2}));
    out3a.backward(Tensor::ones({2}));
    // Don't zero_grad; run another backward.
    auto out3b = src3.gather(Tensor::from_data({0.0, 0.0}, {2}));
    out3b.backward(Tensor::ones({2}));
    // Each backward adds 1.0 → total should be 2.0 in the col-0 cells.
    REQUIRE((*src3.grad())[0] == Approx(2.0));
    REQUIRE((*src3.grad())[2] == Approx(2.0));
}

// ============================================================================
// Section 5 — Gradient accumulation and zero_grad
// ============================================================================

TEST_CASE("Gradient accumulation: multiple backward() calls without zero_grad sum", "[tensor][autograd]") {
    auto x = Tensor::from_data({2.0, 3.0}, {2});
    x.requires_grad_(true);

    // Each backward call adds 1/N to every element (mean backward).
    // After 3 calls: grad should be 3 * 0.5 = 1.5 per element.
    for (int i = 0; i < 3; ++i) {
        auto loss = x.mean();
        loss.backward();
    }

    REQUIRE(x.grad() != nullptr);
    REQUIRE((*x.grad())[0] == Approx(1.5));
    REQUIRE((*x.grad())[1] == Approx(1.5));
}

TEST_CASE("zero_grad() resets gradient to zero", "[tensor][autograd]") {
    auto x = Tensor::from_data({1.0, 2.0}, {2});
    x.requires_grad_(true);

    auto loss = x.mean();
    loss.backward();
    REQUIRE((*x.grad())[0] == Approx(0.5));

    x.zero_grad();
    REQUIRE((*x.grad())[0] == Approx(0.0));
    REQUIRE((*x.grad())[1] == Approx(0.0));
}

// ============================================================================
// Section 6 — Diamond dependency (CRITICAL)
// ============================================================================

TEST_CASE("Diamond dependency: gradient is the correct sum from both paths", "[tensor][autograd]") {
    // Computational graph:
    //
    //       x
    //      / \
    //     a   b       a = x * 2.0,  b = x * 3.0
    //      \ /
    //       c         c = a + b = x*2 + x*3 = x*5
    //       |
    //     loss = mean(c)
    //
    // Analytic gradient:
    //   d(loss)/dx = d(mean(a+b))/dx
    //              = (1/N) * (d(a)/dx + d(b)/dx)
    //              = (1/N) * (2 + 3) = 5/N
    // With x shape [4]: grad = 5/4 = 1.25 per element.

    auto x = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {4});
    x.requires_grad_(true);

    auto a = x.mul(2.0);   // a = 2x
    auto b = x.mul(3.0);   // b = 3x
    auto c = a.add(b);     // c = 5x  — x used via two paths
    auto loss = c.mean();  // scalar

    loss.backward();

    REQUIRE(x.grad() != nullptr);
    for (int64_t i = 0; i < 4; ++i) {
        // Must be 5/4 = 1.25, not just 2/4 or 3/4 from a single path.
        REQUIRE((*x.grad())[i] == Approx(1.25));
    }
}

TEST_CASE("Diamond dependency: deeper graph with two matmul consumers", "[tensor][autograd]") {
    // W is used in both W@x1 and W@x2; loss = mean((W@x1).square()) + mean((W@x2).square())
    // Gradient w.r.t. W must be the sum of contributions from both branches.
    auto W  = Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2});  // identity
    auto x1 = Tensor::from_data({1.0, 2.0}, {2, 1});
    auto x2 = Tensor::from_data({3.0, 4.0}, {2, 1});
    W.requires_grad_(true);

    auto y1   = W.matmul(x1);       // [2,1]
    auto y2   = W.matmul(x2);       // [2,1]
    auto sq1  = y1.square();
    auto sq2  = y2.square();
    auto loss = sq1.mean().add(sq2.mean());  // scalar + scalar → scalar (shape {})

    loss.backward();
    REQUIRE(W.grad() != nullptr);

    // Numerical check to verify the sum from both paths is correct.
    auto xW = Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2});
    auto num = numerical_gradient(xW, [&](Tensor ww) -> Tensor {
        auto yy1 = ww.matmul(x1);
        auto yy2 = ww.matmul(x2);
        return yy1.square().mean().add(yy2.square().mean());
    });

    for (int64_t i = 0; i < 4; ++i) {
        const double analytic = (*W.grad())[i];
        const double numeric  = num[i];
        const double denom    = std::max(std::abs(analytic), std::abs(numeric));
        const double rel_err  = (denom < 1e-8) ? std::abs(analytic - numeric)
                                                 : std::abs(analytic - numeric) / denom;
        REQUIRE(rel_err < kNumGradTol);
    }
}

// ============================================================================
// Section 7 — Numerical gradient checks (central difference) for every op
// ============================================================================

TEST_CASE("Numerical grad check: add", "[tensor][numgrad]") {
    auto x = Tensor::from_data({1.0, -2.0, 3.0}, {3});
    auto y = Tensor::from_data({0.5, 1.5, -1.0}, {3});
    check_grad(x, [&y](Tensor xc) { return xc.add(y).mean(); }, "add");
}

TEST_CASE("Numerical grad check: sub", "[tensor][numgrad]") {
    auto x = Tensor::from_data({2.0, 0.0, -1.0}, {3});
    auto y = Tensor::from_data({1.0, -1.0, 2.0}, {3});
    check_grad(x, [&y](Tensor xc) { return xc.sub(y).mean(); }, "sub");
}

TEST_CASE("Numerical grad check: mul (elementwise)", "[tensor][numgrad]") {
    auto x = Tensor::from_data({2.0, 3.0, -1.0, 4.0}, {4});
    auto y = Tensor::from_data({-1.0, 2.0, 3.0, -2.0}, {4});
    check_grad(x, [&y](Tensor xc) { return xc.mul(y).mean(); }, "mul_elementwise");
}

TEST_CASE("Numerical grad check: mul (scalar)", "[tensor][numgrad]") {
    auto x = Tensor::from_data({1.0, -2.0, 3.0}, {3});
    check_grad(x, [](Tensor xc) { return xc.mul(3.7).mean(); }, "mul_scalar");
}

TEST_CASE("Numerical grad check: matmul", "[tensor][numgrad]") {
    // Gradient w.r.t. A in A @ B → mean
    auto A = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});
    auto B = Tensor::from_data({1.0, -1.0, 2.0, 0.5, -0.5, 1.5}, {3, 2});
    check_grad(A, [&B](Tensor Ac) { return Ac.matmul(B).mean(); }, "matmul_wrtA");

    // Gradient w.r.t. B
    check_grad(B, [&A](Tensor Bc) { return A.matmul(Bc).mean(); }, "matmul_wrtB");
}

TEST_CASE("Numerical grad check: relu (away from zero)", "[tensor][numgrad]") {
    // Avoid points near zero where subgradient makes numerical check ill-defined.
    auto x = Tensor::from_data({1.0, 2.0, 3.0, -1.0, -2.0}, {5});
    check_grad(x, [](Tensor xc) { return xc.relu().mean(); }, "relu");
}

TEST_CASE("Numerical grad check: square", "[tensor][numgrad]") {
    auto x = Tensor::from_data({1.0, -2.0, 3.0, 0.5}, {4});
    check_grad(x, [](Tensor xc) { return xc.square().mean(); }, "square");
}

TEST_CASE("Numerical grad check: mean", "[tensor][numgrad]") {
    auto x = Tensor::from_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {6});
    check_grad(x, [](Tensor xc) { return xc.mean(); }, "mean");
}

TEST_CASE("Numerical grad check: gather", "[tensor][numgrad]") {
    // 3 rows, 4 cols; indices select one column per row.
    auto src = Tensor::from_data({1.0, 2.0, 3.0, 4.0,
                                   5.0, 6.0, 7.0, 8.0,
                                   9.0, 10.0, 11.0, 12.0}, {3, 4});
    auto idx = Tensor::from_data({1.0, 3.0, 0.0}, {3});
    check_grad(src, [&idx](Tensor xc) { return xc.gather(idx).mean(); }, "gather");
}

// ============================================================================
// Section 8 — End-to-end integration: loss = mean((W @ x - target)^2)
// ============================================================================

TEST_CASE("End-to-end: loss = mean((W @ x - target)^2) — gradient via backward and numcheck",
          "[tensor][integration]") {
    // Small linear regression step:
    //   W [2,3], x [3,1], target [2,1]
    //   y = W @ x           [2,1]
    //   diff = y - target   [2,1]
    //   loss = mean(diff^2) scalar
    //
    // We verify dL/dW numerically.

    auto W      = Tensor::from_data({0.5, -0.3, 0.8,
                                      0.1,  0.4, -0.6}, {2, 3});
    auto x      = Tensor::from_data({1.0, -2.0, 3.0}, {3, 1});
    auto target = Tensor::from_data({1.0, -1.0}, {2, 1});

    W.requires_grad_(true);

    // Forward
    auto y    = W.matmul(x);
    auto diff = y.sub(target);
    auto loss = diff.square().mean();

    // Backward
    loss.backward();
    REQUIRE(W.grad() != nullptr);

    // Numerical check for dL/dW
    auto W0 = Tensor::from_data({0.5, -0.3, 0.8, 0.1, 0.4, -0.6}, {2, 3});
    auto num_dW = numerical_gradient(W0, [&](Tensor Wc) {
        return Wc.matmul(x).sub(target).square().mean();
    });

    for (int64_t i = 0; i < W.numel(); ++i) {
        const double analytic = (*W.grad())[i];
        const double numeric  = num_dW[i];
        const double denom    = std::max(std::abs(analytic), std::abs(numeric));
        const double rel_err  = (denom < 1e-8) ? std::abs(analytic - numeric)
                                                 : std::abs(analytic - numeric) / denom;
        INFO("W grad at index " << i << ": analytic=" << analytic
             << " numeric=" << numeric << " rel_err=" << rel_err);
        REQUIRE(rel_err < kNumGradTol);
    }

    // Also verify the loss value matches manual computation.
    // y = W @ x = [0.5*1 + -0.3*(-2) + 0.8*3, 0.1*1 + 0.4*(-2) + -0.6*3]
    //           = [0.5 + 0.6 + 2.4, 0.1 - 0.8 - 1.8]
    //           = [3.5, -2.5]
    // diff = [3.5-1, -2.5-(-1)] = [2.5, -1.5]
    // loss = mean([6.25, 2.25]) = 4.25
    REQUIRE(loss.item() == Approx(4.25));
}

// ============================================================================
// Section 9 — backward() on non-scalar without explicit grad throws
// ============================================================================

TEST_CASE("backward() on non-scalar tensor without explicit grad throws", "[tensor][autograd]") {
    auto x = Tensor::from_data({1.0, 2.0}, {2});
    x.requires_grad_(true);
    auto y = x.square();
    REQUIRE_THROWS_AS(y.backward(), std::invalid_argument);
    // With explicit grad of the same shape — no throw.
    REQUIRE_NOTHROW(y.backward(Tensor::ones({2})));
}
