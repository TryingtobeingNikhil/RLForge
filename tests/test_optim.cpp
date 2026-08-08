// tests/test_optim.cpp — Milestone 6 Part F: SGD, Adam, Optimizer tests.
//
// Includes the critical integration test: after an optimizer step() mutates a
// parameter, calling backward() on the stale pre-step graph throws.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "rl/tensor/tensor.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/optim/optimizer.hpp"
#include "rl/optim/sgd.hpp"
#include "rl/optim/adam.hpp"

using Catch::Approx;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ============================================================================
// SGD Test 1: One step, no momentum — values match hand computation.
//
// param = [1.0, 2.0, 3.0], grad = [0.1, 0.2, 0.3], lr = 0.5
// Expected after one step: param -= 0.5 * grad = [0.95, 1.9, 2.85]
// ============================================================================
TEST_CASE("SGD: one step without momentum matches hand computation", "[optim][sgd]") {
    auto p = std::make_shared<Tensor>(Tensor::from_data({1.0, 2.0, 3.0}, {3}));
    p->requires_grad_(true);

    // Manually set gradient (simulating a backward pass result).
    // accumulate_grad adds to the grad buffer.
    auto g_tensor = Tensor::from_data({0.1, 0.2, 0.3}, {3});
    p->accumulate_grad(g_tensor);

    rl::optim::SGD sgd({p}, /*lr=*/0.5);
    sgd.step();

    REQUIRE((*p)[0] == Approx(1.0 - 0.5 * 0.1));
    REQUIRE((*p)[1] == Approx(2.0 - 0.5 * 0.2));
    REQUIRE((*p)[2] == Approx(3.0 - 0.5 * 0.3));
}

// ============================================================================
// SGD Test 2: Two steps with momentum — confirms velocity accumulation.
//
// param = [1.0], grad step 1 = [0.4], grad step 2 = [0.2]
// lr = 0.1, momentum = 0.9
//
// Step 1: v = 0.9*0 + 0.4 = 0.4       param = 1.0 - 0.1*0.4 = 0.96
// Step 2: v = 0.9*0.4 + 0.2 = 0.56    param = 0.96 - 0.1*0.56 = 0.904
// ============================================================================
TEST_CASE("SGD: two steps with momentum matches hand computation", "[optim][sgd]") {
    auto p = std::make_shared<Tensor>(Tensor::from_data({1.0}, {1}));
    p->requires_grad_(true);

    rl::optim::SGD sgd({p}, /*lr=*/0.1, /*momentum=*/0.9);

    // Step 1
    p->accumulate_grad(Tensor::from_data({0.4}, {1}));
    sgd.step();
    REQUIRE((*p)[0] == Approx(0.96));

    // Step 2 (accumulate new gradient, zero first to simulate per-step zero_grad)
    sgd.zero_grad();
    p->accumulate_grad(Tensor::from_data({0.2}, {1}));
    sgd.step();
    // v = 0.9 * 0.4 + 0.2 = 0.56; param = 0.96 - 0.1 * 0.56 = 0.904
    REQUIRE((*p)[0] == Approx(0.904));
}

// ============================================================================
// Adam Test: One step — values match hand computation with bias correction.
//
// param = [1.0], grad = [0.5]
// lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, t=1
//
// m = 0.9*0 + 0.1*0.5 = 0.05
// v = 0.999*0 + 0.001*0.25 = 0.00025
// m_hat = 0.05 / (1 - 0.9) = 0.5
// v_hat = 0.00025 / (1 - 0.999) = 0.25
// update = 0.001 * 0.5 / (sqrt(0.25) + 1e-8) ≈ 0.001 * 0.5 / 0.5 = 0.001
// param = 1.0 - 0.001 = 0.999
// ============================================================================
TEST_CASE("Adam: one step matches hand computation with bias correction", "[optim][adam]") {
    auto p = std::make_shared<Tensor>(Tensor::from_data({1.0}, {1}));
    p->requires_grad_(true);
    p->accumulate_grad(Tensor::from_data({0.5}, {1}));

    rl::optim::Adam adam({p});  // default hyperparams
    adam.step();

    const double m     = 0.1 * 0.5;                      // 0.05
    const double v     = 0.001 * 0.25;                   // 0.00025
    const double bc1   = 1.0 - 0.9;                      // 0.1
    const double bc2   = 1.0 - 0.999;                    // 0.001
    const double m_hat = m / bc1;                         // 0.5
    const double v_hat = v / bc2;                         // 0.25
    const double upd   = 0.001 * m_hat / (std::sqrt(v_hat) + 1e-8);
    const double expected = 1.0 - upd;

    REQUIRE((*p)[0] == Approx(expected).epsilon(1e-9));
}

// ============================================================================
// zero_grad Test: confirms all parameters' grad buffers are zeroed.
// ============================================================================
TEST_CASE("Optimizer::zero_grad resets all parameter gradients", "[optim]") {
    auto p1 = std::make_shared<Tensor>(Tensor::zeros({3}));
    p1->requires_grad_(true);
    p1->accumulate_grad(Tensor::from_data({1.0, 2.0, 3.0}, {3}));

    auto p2 = std::make_shared<Tensor>(Tensor::zeros({2}));
    p2->requires_grad_(true);
    p2->accumulate_grad(Tensor::from_data({4.0, 5.0}, {2}));

    rl::optim::SGD sgd({p1, p2}, 0.01);

    // Confirm gradients are set before zeroing.
    REQUIRE(p1->grad() != nullptr);
    REQUIRE(p2->grad() != nullptr);

    sgd.zero_grad();

    REQUIRE(p1->grad() != nullptr);  // buffer still exists, just zeroed
    REQUIRE(p2->grad() != nullptr);
    for (int64_t i = 0; i < p1->numel(); ++i) {
        REQUIRE((*p1->grad())[i] == Approx(0.0));
    }
    for (int64_t i = 0; i < p2->numel(); ++i) {
        REQUIRE((*p2->grad())[i] == Approx(0.0));
    }
}

// ============================================================================
// CRITICAL Integration Test: After optimizer step() mutates a parameter,
// calling backward() on the STALE graph from the pre-step forward pass
// throws via the Part A version guard.
//
// This proves that Parts A and F work together correctly in the real usage
// pattern the version guard was designed for.
// ============================================================================
TEST_CASE("Version guard integration: backward on stale graph after optimizer step throws",
          "[optim][version_guard]") {
    // Build a simple computation: loss = mean(w^2)
    // w is a parameter that the optimizer will mutate.
    auto w = std::make_shared<Tensor>(Tensor::from_data({1.0, 2.0, 3.0}, {3}));
    w->requires_grad_(true);

    // Forward pass — this builds the autograd graph and captures w's Storage
    // version (currently 0, since no mutations have happened yet).
    auto loss = w->mul(*w).mean();

    // Run backward (first call — should work fine, no mutation yet).
    loss.backward();
    REQUIRE(w->grad() != nullptr);

    // Now simulate what happens in a real training loop:
    // the optimizer mutates w in-place via data_mutable() (bumps version to 1+).
    rl::optim::SGD sgd({w}, 0.1);
    sgd.step();  // w.data_mutable() called internally — version bumped

    // Reset gradients and try to backward() through the STALE graph.
    // The backward closure for mul captured version=0 at forward time.
    // w's Storage version is now >0 (bumped by the optimizer step).
    // This should throw std::runtime_error via the version guard.
    w->zero_grad();
    REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
}
