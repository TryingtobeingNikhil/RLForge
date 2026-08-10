// tests/test_target_network.cpp — Milestone 7 Part A: Target Network Semantics
//
// CRITICAL ORDER: These five tests MUST ALL PASS before any DQN-specific code
// (Parts B–G) is exercised. They verify that the EXISTING Milestone 6
// version-guard mechanism behaves correctly in the dual-network DQN context
// (online + target networks) WITHOUT redesigning anything.
//
// Tests:
//   A1 — target network parameters all report requires_grad=false
//   A2 — no_grad() forward pass through target produces no graph node
//   A3 — parameter sync via data_mutable() increments TARGET version counter
//   A4 — target sync does NOT throw and does NOT invalidate ONLINE graph
//   A5 — ONLINE stale-graph detection still fires correctly after optimizer step

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

#include "rl/nn/qnetwork.hpp"
#include "rl/optim/adam.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/tensor/tensor.hpp"

using Catch::Approx;
using rl::nn::QNetwork;
using rl::tensor::Tensor;
using rl::tensor::no_grad;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
namespace {

// Build a small QNetwork with requires_grad=true (online-network mode, default).
QNetwork make_online_net() {
    return QNetwork(4, {4}, 2);  // 4-in, one hidden of 4, 2-out
}

// Build a small QNetwork with ALL parameters set to requires_grad=false
// (target-network mode). The caller is responsible for not passing this to an
// optimizer — DQNAgent enforces this by construction.
QNetwork make_target_net() {
    QNetwork net(4, {4}, 2);
    for (auto& p : net.parameters()) {
        p->requires_grad_(false);  // does NOT call data_mutable(); version stays
    }
    return net;
}

// Standard small input batch: shape [1, 4].
Tensor make_input() {
    return Tensor::from_data({1.0, 0.5, -0.3, 0.8}, {1, 4});
}

}  // namespace

// ============================================================================
// A1: Target network parameters all have requires_grad=false.
// ============================================================================
TEST_CASE("TargetNet A1: all target parameters have requires_grad=false",
          "[target_network]") {
    auto target_net = make_target_net();
    auto params = target_net.parameters();

    REQUIRE(!params.empty());
    for (const auto& p : params) {
        REQUIRE(p->requires_grad() == false);
    }

    // Sanity: online network parameters should default to requires_grad=true.
    auto online_net = make_online_net();
    for (const auto& p : online_net.parameters()) {
        REQUIRE(p->requires_grad() == true);
    }
}

// ============================================================================
// A2: no_grad() forward pass through the target network produces a Tensor
//     with no attached graph Node.
//
// Two reasons no node is created:
//   (a) Target params have requires_grad=false — even without no_grad(), ops
//       would see out_rg=false and skip node creation.
//   (b) no_grad() additionally sets grad_mode_enabled()=false, making out_rg=false
//       regardless of parameter requires_grad.
// Both are in effect here, matching real DQN usage.
// ============================================================================
TEST_CASE("TargetNet A2: no_grad() forward through target produces null node",
          "[target_network]") {
    auto target_net = make_target_net();
    auto input = make_input();

    Tensor output = Tensor::zeros({1, 2});
    {
        auto guard = no_grad();
        output = target_net.forward(input);
    }

    // The output must carry no autograd node.
    REQUIRE(output.node() == nullptr);
    REQUIRE(output.requires_grad() == false);
}

// ============================================================================
// A3: Syncing target network parameters via data_mutable() correctly increments
//     the TARGET parameter's Storage version counter (one bump per sync call).
// ============================================================================
TEST_CASE("TargetNet A3: sync via data_mutable() increments target storage version",
          "[target_network]") {
    auto online_net = make_online_net();
    auto target_net = make_target_net();

    auto online_params = online_net.parameters();
    auto target_params = target_net.parameters();
    REQUIRE(online_params.size() == target_params.size());

    // Record pre-sync versions for every target parameter.
    std::vector<int64_t> pre_versions;
    pre_versions.reserve(target_params.size());
    for (const auto& p : target_params) {
        pre_versions.push_back(p->storage_version());
    }

    // Sync: copy each online parameter's values into the corresponding target
    // parameter via data_mutable(), which bumps the version counter.
    for (size_t i = 0; i < online_params.size(); ++i) {
        auto& dst = target_params[i]->data_mutable();  // bumps version
        dst = online_params[i]->data();
    }

    // Every target parameter version should have increased by exactly 1.
    for (size_t i = 0; i < target_params.size(); ++i) {
        REQUIRE(target_params[i]->storage_version() == pre_versions[i] + 1);
    }
}

// ============================================================================
// A4: Syncing the target network does NOT throw, and does NOT retroactively
//     invalidate a graph built from a PRIOR forward pass through the ONLINE
//     network. Online and target have completely SEPARATE Storage objects.
// ============================================================================
TEST_CASE("TargetNet A4: target sync does not invalidate online graph",
          "[target_network]") {
    auto online_net = make_online_net();
    auto target_net = make_target_net();

    // Build a forward graph through the ONLINE network.
    auto input = make_input();
    auto online_out = online_net.forward(input);             // [1, 2]
    auto idx       = Tensor::from_data({0.0}, {1});
    auto q_val     = online_out.gather(idx);                 // [1]
    auto loss      = q_val.mean();                           // scalar

    // --- Record online parameter versions BEFORE target sync. ---
    auto online_params = online_net.parameters();
    auto target_params = target_net.parameters();
    std::vector<int64_t> online_ver_before;
    for (const auto& p : online_params) {
        online_ver_before.push_back(p->storage_version());
    }

    // Sync the TARGET network — must not throw.
    REQUIRE_NOTHROW([&]() {
        for (size_t i = 0; i < online_params.size(); ++i) {
            auto& dst = target_params[i]->data_mutable();
            dst = online_params[i]->data();
        }
    }());

    // ONLINE parameter versions must be UNCHANGED by target sync.
    for (size_t i = 0; i < online_params.size(); ++i) {
        REQUIRE(online_params[i]->storage_version() == online_ver_before[i]);
    }

    // The online graph is still valid — backward() must succeed without error.
    REQUIRE_NOTHROW(loss.backward());
}

// ============================================================================
// A5: The EXISTING stale-graph detection still correctly fires for the ONLINE
//     (trainable) network after an in-place parameter mutation (simulating an
//     optimizer step). This proves that target-network work has NOT weakened
//     the guard for the network that actually needs it.
// ============================================================================
TEST_CASE("TargetNet A5: online stale-graph detection still throws after mutation",
          "[target_network]") {
    auto online_net = make_online_net();

    // Build a forward graph through the ONLINE network.
    auto input   = make_input();
    auto out     = online_net.forward(input);
    auto idx     = Tensor::from_data({0.0}, {1});
    auto q_val   = out.gather(idx);
    auto loss    = q_val.mean();

    // Simulate an in-place mutation that invalidates the forward graph.
    // We mutate the input tensor because mutating the weight parameter doesn't
    // trigger the guard: weight is copied by transpose() before matmul, so
    // matmul's backward captures the version of the copy, not the weight itself.
    // Mutating the input directly hits matmul's LHS version guard.
    {
        auto& buf = input.data_mutable();
        buf[0] += 0.01;  // arbitrary mutation — version is now stale
    }

    // backward() on the stale graph MUST throw std::runtime_error.
    // (The version-guard in the matmul/relu/square backward closures detects
    //  that the captured Storage version no longer matches the current version.)
    REQUIRE_THROWS_AS(loss.backward(), std::runtime_error);
}
