// tests/test_version_guard.cpp — Milestone 6 Part A: Version-counter guard
//
// Tests that in-place mutation of a tensor after a forward pass has been
// built into an autograd graph is correctly detected by the version counter,
// and that normal (non-mutating) use paths produce no false positives.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <stdexcept>
#include <string>

#include "rl/tensor/tensor.hpp"
#include "rl/tensor/autograd.hpp"

using Catch::Approx;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ============================================================================
// Test 1: Mutating a tensor AFTER forward, THEN calling backward() throws.
//
// This is the primary use case the version guard was designed to catch:
// an optimizer mutates a parameter in-place after the forward graph has been
// built but before backward() has been called.
// ============================================================================
TEST_CASE("Version guard: in-place mutation after forward pass throws on backward",
          "[version_guard]") {
    // Build: loss = mean(x * x)
    auto x = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    x.requires_grad_(true);

    auto loss = x.mul(x).mean();  // squares then mean

    // Mutate x in place AFTER the forward graph is built.
    // data_mutable() bumps the version counter, making the saved version stale.
    auto& buf = x.data_mutable();
    buf[0] = 99.0;  // mutation

    // backward() should detect the stale version in the square/mul backward.
    REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);

    // Confirm the error message identifies the stale op.
    try {
        // Re-build to throw again (backward consumes the graph state).
        auto x2 = Tensor::from_data({1.0, 2.0, 3.0}, {3});
        x2.requires_grad_(true);
        auto loss2 = x2.mul(x2).mean();
        auto& buf2 = x2.data_mutable();
        buf2[0] = 99.0;
        loss2.backward();
        FAIL("Expected std::runtime_error but no exception was thrown");
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        // Should mention "backward" and "mutated in-place"
        REQUIRE(msg.find("backward") != std::string::npos);
        REQUIRE(msg.find("mutated") != std::string::npos);
    }
}

// ============================================================================
// Test 2: Normal use (no mutation) does NOT throw and produces correct gradients.
//
// Confirms the guard produces no false positives on the happy path.
// ============================================================================
TEST_CASE("Version guard: normal use without mutation does not throw and is correct",
          "[version_guard]") {
    // loss = mean(x * x),  d(loss)/dx[i] = 2*x[i] / n
    auto x = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    x.requires_grad_(true);

    // No in-place mutation between forward and backward.
    REQUIRE_NOTHROW([&]() {
        auto loss = x.mul(x).mean();
        loss.backward();
    }());

    REQUIRE(x.grad() != nullptr);
    // d/dx mean(x^2) = 2x/3
    REQUIRE((*x.grad())[0] == Approx(2.0 / 3.0));
    REQUIRE((*x.grad())[1] == Approx(4.0 / 3.0));
    REQUIRE((*x.grad())[2] == Approx(2.0));
}

// ============================================================================
// Test 3: Mutating a tensor BEFORE using it in a forward pass does NOT
//         produce a false positive.
//
// The version is captured at closure-creation time (forward pass), not at
// object-creation time. So mutations BEFORE the forward pass are fine.
// ============================================================================
TEST_CASE("Version guard: mutation before forward pass does not cause false positive",
          "[version_guard]") {
    auto x = Tensor::from_data({1.0, 2.0, 3.0}, {3});
    x.requires_grad_(true);

    // Mutate BEFORE the forward pass — version bumps here.
    {
        auto& buf = x.data_mutable();
        buf[0] = 4.0;
        buf[1] = 5.0;
        buf[2] = 6.0;
    }
    // Version is now 1. The forward pass below will capture version=1.
    // No further mutations between forward and backward.

    // loss = mean(x * x) with x = [4,5,6]
    REQUIRE_NOTHROW([&]() {
        x.zero_grad();
        auto loss = x.mul(x).mean();
        loss.backward();
    }());

    REQUIRE(x.grad() != nullptr);
    // d/dx mean(x^2) = 2x/3, with x=[4,5,6]
    REQUIRE((*x.grad())[0] == Approx(8.0 / 3.0));
    REQUIRE((*x.grad())[1] == Approx(10.0 / 3.0));
    REQUIRE((*x.grad())[2] == Approx(4.0));
}

// ============================================================================
// Test 4: Mutation of an intermediate (non-leaf) tensor is also caught.
//
// Intermediate tensors are produced by ops. Mutating them in-place after they
// have been captured into a downstream backward closure should be detected.
// ============================================================================
TEST_CASE("Version guard: mutation of intermediate tensor caught in multi-op graph",
          "[version_guard]") {
    // Build: y = relu(x), loss = mean(y * y)
    // y is an intermediate tensor. Mutating y before backward() should be caught
    // by the mul backward closure when it checks y's storage version.
    auto x = Tensor::from_data({-1.0, 2.0, 3.0}, {3});
    x.requires_grad_(true);

    auto y    = x.relu();     // intermediate: y = max(0,x) = [0,2,3]
    auto loss = y.mul(y).mean();

    // Mutate y in-place AFTER it was captured by the mul backward closure.
    {
        auto& buf = y.data_mutable();
        buf[0] = 99.0;  // stale mutation
    }

    // The mul backward will check y's storage version — should throw.
    REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
}
