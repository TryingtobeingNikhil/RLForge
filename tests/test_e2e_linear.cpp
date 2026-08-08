// tests/test_e2e_linear.cpp — Milestone 6 Part G: End-to-end training loop.
//
// Smoke test: fit y = 2x + 1 with a single Linear(1,1) layer using SGD.
// Verifies that the loss decreases across training iterations, confirming
// that the full forward→loss→backward→step pipeline works end-to-end.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

#include "rl/tensor/tensor.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/nn/linear.hpp"
#include "rl/nn/losses.hpp"
#include "rl/optim/sgd.hpp"
#include "rl/optim/adam.hpp"

using Catch::Approx;
using rl::tensor::Tensor;

// ============================================================================
// End-to-end test: fit y = 2x + 1 using Linear(1,1) + SGD.
//
// Training data: x in [-1, 0, 1], y_target = 2x + 1 = [-1, 1, 3].
// The loss should decrease monotonically (or at least overall) across 100
// training steps.
//
// This exercises the full Milestone 6 pipeline:
//   - Linear forward (matmul + transpose + broadcasting add)
//   - MSE loss (sub + square + mean)
//   - backward() through the full graph
//   - SGD step() (in-place mutation with version counter bump)
//   - zero_grad() before each iteration
// ============================================================================
TEST_CASE("End-to-end: Linear(1,1) fits y=2x+1 with SGD, loss decreases",
          "[integration][e2e]") {
    rl::nn::Linear layer(1, 1);
    auto params = layer.parameters();
    rl::optim::SGD optimizer(params, /*lr=*/0.05);

    // Synthetic training data: y = 2x + 1
    // Input: [3,1], Target: [3,1]
    auto x_data = Tensor::from_data({-1.0, 0.0, 1.0}, {3, 1});
    auto y_target = Tensor::from_data({-1.0, 1.0, 3.0}, {3, 1});

    double first_loss = -1.0;
    double last_loss  = -1.0;

    constexpr int kSteps = 100;
    for (int step = 0; step < kSteps; ++step) {
        // Forward pass.
        auto y_pred = layer.forward(x_data);

        // Loss.
        auto loss = rl::nn::mse_loss(y_pred, y_target);
        const double loss_val = loss.item();

        if (step == 0)           first_loss = loss_val;
        if (step == kSteps - 1)  last_loss  = loss_val;

        // Backward pass.
        optimizer.zero_grad();
        loss.backward();

        // Parameter update.
        optimizer.step();
    }

    INFO("First loss: " << first_loss << ", Last loss: " << last_loss);
    REQUIRE(first_loss >= 0.0);
    REQUIRE(last_loss  >= 0.0);
    // Loss must decrease significantly over 100 steps.
    REQUIRE(last_loss < first_loss * 0.1);
}

// ============================================================================
// End-to-end test: same regression, but using Adam optimizer.
// ============================================================================
TEST_CASE("End-to-end: Linear(1,1) fits y=2x+1 with Adam, loss decreases",
          "[integration][e2e]") {
    rl::nn::Linear layer(1, 1);
    auto params = layer.parameters();
    rl::optim::Adam optimizer(params, /*lr=*/0.05);

    auto x_data   = Tensor::from_data({-1.0, 0.0, 1.0}, {3, 1});
    auto y_target = Tensor::from_data({-1.0, 1.0, 3.0}, {3, 1});

    double first_loss = -1.0;
    double last_loss  = -1.0;

    constexpr int kSteps = 100;
    for (int step = 0; step < kSteps; ++step) {
        auto y_pred  = layer.forward(x_data);
        auto loss    = rl::nn::mse_loss(y_pred, y_target);
        const double loss_val = loss.item();

        if (step == 0)           first_loss = loss_val;
        if (step == kSteps - 1)  last_loss  = loss_val;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    INFO("First loss (Adam): " << first_loss << ", Last loss (Adam): " << last_loss);
    REQUIRE(first_loss >= 0.0);
    REQUIRE(last_loss  >= 0.0);
    REQUIRE(last_loss < first_loss * 0.1);
}
