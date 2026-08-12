#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "rl/tensor/tensor.hpp"

using Catch::Approx;
using rl::tensor::Tensor;

TEST_CASE("PPO tensor ops have stable forward values", "[ppo][tensor]") {
    auto logits = Tensor::from_data({1000.0, 1001.0, -1000.0, -1000.0}, {2, 2});
    auto log_probs = logits.log_softmax();
    REQUIRE(std::exp(log_probs[0]) + std::exp(log_probs[1]) == Approx(1.0));
    REQUIRE(std::exp(log_probs[2]) + std::exp(log_probs[3]) == Approx(1.0));

    auto ratio = Tensor::from_data({0.5, 1.0, 1.5}, {3});
    auto clipped = ratio.clamp(0.8, 1.2);
    REQUIRE(clipped[0] == Approx(0.8));
    REQUIRE(clipped[1] == Approx(1.0));
    REQUIRE(clipped[2] == Approx(1.2));

    auto minimum = ratio.minimum(Tensor::from_data({0.7, 0.9, 2.0}, {3}));
    REQUIRE(minimum[0] == Approx(0.5));
    REQUIRE(minimum[1] == Approx(0.9));
    REQUIRE(minimum[2] == Approx(1.5));
}

TEST_CASE("log_softmax backward sums to zero across each row", "[ppo][tensor][autograd]") {
    auto logits = Tensor::from_data({0.2, -0.1, 0.7, 1.0, -2.0, 0.5}, {2, 3});
    logits.requires_grad_(true);
    auto log_probs = logits.log_softmax();
    log_probs.backward(Tensor::ones({2, 3}));

    REQUIRE(logits.grad() != nullptr);
    for (int64_t row = 0; row < 2; ++row) {
        double row_sum = 0.0;
        for (int64_t col = 0; col < 3; ++col) {
            row_sum += (*logits.grad())[row * 3 + col];
        }
        REQUIRE(row_sum == Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("exp and log compose with correct gradient", "[ppo][tensor][autograd]") {
    auto x = Tensor::from_data({0.5, 1.5, 2.0}, {3});
    x.requires_grad_(true);
    x.exp().log().mean().backward();
    for (int64_t i = 0; i < 3; ++i) {
        REQUIRE((*x.grad())[i] == Approx(1.0 / 3.0));
    }
}
