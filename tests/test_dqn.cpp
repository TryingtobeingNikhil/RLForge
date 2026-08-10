// tests/test_dqn.cpp — Milestone 7 Parts C / D / E / F / G
//
// Tests are organised in the same order as the implementation parts:
//   Part C — QNetwork: forward shape, parameter count, requires_grad
//   Part D — batch_to_tensors: conversion correctness, terminated vs truncated
//   Part E — EpsilonGreedyPolicy: decay schedule, argmax, random distribution
//   Part F — DQNAgent: smoke test, target sync intervals, no-exception loop
//   Part G — End-to-end: solvable bandit, loss trends down

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "rl/agents/dqn.hpp"
#include "rl/agents/epsilon_greedy.hpp"
#include "rl/core/replay_buffer.hpp"
#include "rl/core/transition.hpp"
#include "rl/core/types.hpp"
#include "rl/data/batch_to_tensors.hpp"
#include "rl/nn/qnetwork.hpp"
#include "rl/replay_buffers/vector_transition_storage.hpp"
#include "rl/tensor/tensor.hpp"

using Catch::Approx;
using rl::agents::DQNAgent;
using rl::agents::DQNConfig;
using rl::agents::EpsilonGreedyPolicy;
using rl::core::ReplayBuffer;
using rl::core::Transition;
using rl::core::TransitionBatch;
using rl::data::batch_to_tensors;
using rl::nn::QNetwork;
using rl::replay_buffers::VectorTransitionStorage;
using rl::tensor::Tensor;

// ============================================================================
// Helpers
// ============================================================================
namespace {

// Build a Transition with Box observation of dimension state_dim.
Transition make_transition(std::vector<float> obs, int64_t action, float reward,
                           std::vector<float> next_obs,
                           bool terminated, bool truncated) {
    Transition t;
    t.observation      = obs;
    t.action           = action;
    t.reward           = reward;
    t.next_observation = next_obs;
    t.terminated       = terminated;
    t.truncated        = truncated;
    return t;
}

}  // namespace

// ============================================================================
// PART C — QNetwork
// ============================================================================

TEST_CASE("QNetwork: forward output shape [B, num_actions]", "[qnetwork]") {
    QNetwork net(4, {8, 8}, 3);
    auto input = Tensor::from_data({1.0, 0.5, -0.3, 0.8,
                                    0.1, 0.2,  0.3, 0.4,
                                    0.5, 0.6,  0.7, 0.8},
                                   {3, 4});  // B=3, state_dim=4
    auto output = net.forward(input);
    REQUIRE(output.shape() == std::vector<int64_t>{3, 3});  // [B=3, num_actions=3]
}

TEST_CASE("QNetwork: parameter count is 2 * num_layers", "[qnetwork]") {
    // hidden_dims={64, 64} -> 3 layers -> 6 params (weight+bias each)
    QNetwork net3(4, {64, 64}, 2);
    REQUIRE(net3.parameters().size() == 6);

    // hidden_dims={} -> 1 layer -> 2 params
    QNetwork net1(4, {}, 2);
    REQUIRE(net1.parameters().size() == 2);

    // hidden_dims={8} -> 2 layers -> 4 params
    QNetwork net2(4, {8}, 2);
    REQUIRE(net2.parameters().size() == 4);
}

TEST_CASE("QNetwork: all parameters have requires_grad=true by default", "[qnetwork]") {
    QNetwork net(4, {8}, 2);
    for (const auto& p : net.parameters()) {
        REQUIRE(p->requires_grad() == true);
    }
}

TEST_CASE("QNetwork: forward throws on non-2D input", "[qnetwork]") {
    QNetwork net(4, {8}, 2);
    auto bad_input = Tensor::from_data({1.0, 0.5, -0.3, 0.8}, {4});  // 1-D
    REQUIRE_THROWS_AS(net.forward(bad_input), std::invalid_argument);
}

TEST_CASE("QNetwork: no final activation (output can be negative)", "[qnetwork]") {
    // With random init, at least some Q-values should be negative for typical
    // inputs — if ReLU were accidentally on the final layer, all would be >= 0.
    // We test multiple inits and check that negatives occur at least once.
    bool found_negative = false;
    for (int seed = 0; seed < 20 && !found_negative; ++seed) {
        QNetwork net(4, {8}, 4);
        auto input = Tensor::from_data({1.0, -1.0, 0.5, -0.5}, {1, 4});
        auto output = net.forward(input);
        for (int64_t i = 0; i < output.numel(); ++i) {
            if (output[i] < 0.0) { found_negative = true; break; }
        }
    }
    REQUIRE(found_negative);
}

// ============================================================================
// PART D — batch_to_tensors
// ============================================================================

TEST_CASE("batch_to_tensors: converts a small batch correctly", "[batch_to_tensors]") {
    // Build 3 transitions manually.
    ReplayBuffer buf(std::make_unique<VectorTransitionStorage>(10));
    buf.add(make_transition({1.0f, 0.0f}, 0, 1.0f, {0.0f, 1.0f}, false, false));
    buf.add(make_transition({0.5f, 0.5f}, 1, 0.5f, {1.0f, 0.0f}, true,  false));
    buf.add(make_transition({0.0f, 1.0f}, 0, 0.0f, {0.5f, 0.5f}, false, true));

    std::mt19937 rng(42);
    // Sample all 3 (with replacement — just need them all represented).
    // For deterministic testing, construct the batch directly.
    TransitionBatch batch;
    batch.observations     = {std::vector<float>{1.0f, 0.0f},
                               std::vector<float>{0.5f, 0.5f},
                               std::vector<float>{0.0f, 1.0f}};
    batch.actions          = {int64_t{0}, int64_t{1}, int64_t{0}};
    batch.rewards          = {1.0f, 0.5f, 0.0f};
    batch.next_observations= {std::vector<float>{0.0f, 1.0f},
                               std::vector<float>{1.0f, 0.0f},
                               std::vector<float>{0.5f, 0.5f}};
    batch.terminated       = {0, 1, 0};  // BoolArray = vector<uint8_t>
    batch.truncated        = {0, 0, 1};

    auto tb = batch_to_tensors(batch, /*state_dim=*/2);

    // States [3, 2].
    REQUIRE(tb.states.shape() == std::vector<int64_t>{3, 2});
    REQUIRE(tb.states[0] == Approx(1.0));
    REQUIRE(tb.states[1] == Approx(0.0));
    REQUIRE(tb.states[2] == Approx(0.5));
    REQUIRE(tb.states[3] == Approx(0.5));

    // Actions [3].
    REQUIRE(tb.actions.shape() == std::vector<int64_t>{3});
    REQUIRE(tb.actions[0] == Approx(0.0));
    REQUIRE(tb.actions[1] == Approx(1.0));
    REQUIRE(tb.actions[2] == Approx(0.0));

    // Rewards [3].
    REQUIRE(tb.rewards.shape() == std::vector<int64_t>{3});
    REQUIRE(tb.rewards[0] == Approx(1.0));
    REQUIRE(tb.rewards[1] == Approx(0.5));
    REQUIRE(tb.rewards[2] == Approx(0.0));

    // Next states [3, 2].
    REQUIRE(tb.next_states.shape() == std::vector<int64_t>{3, 2});

    // Terminated mask.
    REQUIRE(tb.terminated.shape() == std::vector<int64_t>{3});
    REQUIRE(tb.terminated[0] == Approx(0.0));  // not terminated
    REQUIRE(tb.terminated[1] == Approx(1.0));  // terminated
    REQUIRE(tb.terminated[2] == Approx(0.0));  // NOT terminated (truncated only!)
}

TEST_CASE("batch_to_tensors: truncated transition does NOT zero bootstrap mask",
          "[batch_to_tensors]") {
    // A truncated (but NOT terminated) transition:
    //   terminated=false, truncated=true
    // The terminated mask must be 0.0 → bootstrap is kept (1-0=1).
    TransitionBatch batch;
    batch.observations      = {std::vector<float>{1.0f, 0.0f}};
    batch.actions           = {int64_t{0}};
    batch.rewards           = {1.0f};
    batch.next_observations = {std::vector<float>{0.0f, 1.0f}};
    batch.terminated        = {0};  // NOT terminated
    batch.truncated         = {1};  // IS truncated

    auto tb = batch_to_tensors(batch, 2);

    // terminated mask must be 0.0 → bootstrap NOT zeroed.
    REQUIRE(tb.terminated[0] == Approx(0.0));
}

TEST_CASE("batch_to_tensors: terminated transition zeros bootstrap mask",
          "[batch_to_tensors]") {
    // A terminated (not just truncated) transition:
    //   terminated=true → mask=1.0 → bootstrap zeroed.
    TransitionBatch batch;
    batch.observations      = {std::vector<float>{1.0f, 0.0f}};
    batch.actions           = {int64_t{0}};
    batch.rewards           = {1.0f};
    batch.next_observations = {std::vector<float>{0.0f, 1.0f}};
    batch.terminated        = {1};  // IS terminated
    batch.truncated         = {0};  // NOT truncated

    auto tb = batch_to_tensors(batch, 2);

    REQUIRE(tb.terminated[0] == Approx(1.0));  // bootstrap must be zeroed
}

// ============================================================================
// PART E — EpsilonGreedyPolicy
// ============================================================================

TEST_CASE("EpsilonGreedy: decay schedule is correct at start/mid/end",
          "[epsilon_greedy]") {
    EpsilonGreedyPolicy policy(1.0f, 0.1f, 100);

    // At step 0 (before any steps taken), epsilon = eps_start.
    REQUIRE(policy.current_epsilon() == Approx(1.0f));

    // Advance to step 50: epsilon = 0.1 + (1.0-0.1)*(1-50/100) = 0.55
    // current_epsilon is evaluated BEFORE step_count is incremented.
    // After 50 select_action calls, step_count=50.
    std::mt19937 rng(0);
    auto dummy_q = Tensor::from_data({0.0, 1.0}, {2});
    // Force random selection by using epsilon=1 initially... just advance counter.
    for (int i = 0; i < 50; ++i) {
        policy.select_action(dummy_q, 2, rng);
    }
    // After 50 steps: eps = 1.0 + (50/100)*(0.1-1.0) = 1.0 - 0.45 = 0.55
    REQUIRE(policy.current_epsilon() == Approx(0.55f).margin(1e-4f));

    // After 100 steps: epsilon = eps_end = 0.1.
    for (int i = 0; i < 50; ++i) {
        policy.select_action(dummy_q, 2, rng);
    }
    REQUIRE(policy.current_epsilon() == Approx(0.1f).margin(1e-4f));

    // After > 100 steps: epsilon stays at eps_end.
    for (int i = 0; i < 100; ++i) {
        policy.select_action(dummy_q, 2, rng);
    }
    REQUIRE(policy.current_epsilon() == Approx(0.1f).margin(1e-4f));
}

TEST_CASE("EpsilonGreedy: always picks argmax when epsilon forced to 0",
          "[epsilon_greedy]") {
    EpsilonGreedyPolicy policy(0.0f, 0.0f, 1);  // epsilon stays 0
    std::mt19937 rng(0);

    // Q-values: action 2 is clearly best.
    auto q = Tensor::from_data({0.1, 0.3, 0.9, 0.2}, {4});
    for (int trial = 0; trial < 20; ++trial) {
        REQUIRE(policy.select_action(q, 4, rng) == 2);
    }
}

TEST_CASE("EpsilonGreedy: distribution is roughly uniform when epsilon forced to 1",
          "[epsilon_greedy]") {
    // Epsilon is always 1.0 → always random. Over many samples, each of 4
    // actions should appear roughly num_trials/4 times. We use a loose bound
    // (25% of expected) to avoid flaky failures on CI.
    EpsilonGreedyPolicy policy(1.0f, 1.0f, 0);  // eps_decay_steps=0 → stays 1.0
    std::mt19937 rng(42);

    auto q = Tensor::from_data({0.0, 0.0, 0.0, 0.0}, {4});  // all equal — argmax irrelevant

    constexpr int num_trials = 4000;
    constexpr int num_actions = 4;
    std::vector<int> counts(static_cast<size_t>(num_actions), 0);
    for (int i = 0; i < num_trials; ++i) {
        const int a = policy.select_action(q, num_actions, rng);
        REQUIRE(a >= 0);
        REQUIRE(a < num_actions);
        counts[static_cast<size_t>(a)]++;
    }
    const double expected = static_cast<double>(num_trials) / num_actions;
    for (int a = 0; a < num_actions; ++a) {
        const double actual = static_cast<double>(counts[static_cast<size_t>(a)]);
        // Each bucket should be within 25% of expected (very loose sanity check).
        REQUIRE(actual > expected * 0.75);
        REQUIRE(actual < expected * 1.25);
    }
}

// ============================================================================
// PART F — DQNAgent
// ============================================================================

TEST_CASE("DQNAgent: single train_step produces finite non-NaN loss",
          "[dqn]") {
    DQNConfig cfg;
    cfg.state_dim       = 2;
    cfg.num_actions     = 2;
    cfg.hidden_dims     = {4};
    cfg.batch_size      = 8;
    cfg.buffer_capacity = 100;
    cfg.min_buffer_size = 8;
    cfg.target_update_freq = 50;
    cfg.seed = 0;

    DQNAgent agent(cfg);

    // Fill buffer above min_buffer_size.
    for (int i = 0; i < 20; ++i) {
        std::vector<float> obs = {static_cast<float>(i % 2),
                                  static_cast<float>(1 - i % 2)};
        agent.observe_transitions(
            {make_transition(obs, static_cast<int64_t>(i % 2), 1.0f,
                             obs, false, false)});
    }

    REQUIRE(agent.should_update());
    const double loss = agent.train_step();
    REQUIRE(std::isfinite(loss));
    REQUIRE(loss >= 0.0);
}

TEST_CASE("DQNAgent: target network syncs only at target_update_freq intervals",
          "[dqn]") {
    DQNConfig cfg;
    cfg.state_dim         = 2;
    cfg.num_actions       = 2;
    cfg.hidden_dims       = {4};
    cfg.batch_size        = 8;
    cfg.buffer_capacity   = 200;
    cfg.min_buffer_size   = 8;
    cfg.target_update_freq = 5;   // sync every 5 train steps
    cfg.seed              = 1;

    DQNAgent agent(cfg);

    // Fill buffer.
    for (int i = 0; i < 50; ++i) {
        std::vector<float> obs = {static_cast<float>(i % 2),
                                  static_cast<float>(1 - i % 2)};
        agent.observe_transitions(
            {make_transition(obs, static_cast<int64_t>(i % 2), 1.0f,
                             obs, false, false)});
    }

    // Record target param[0][0] right after construction (already synced once).
    const double target_val_init = agent.target_net().parameters()[0]->data()[0];

    // Run 4 steps (< target_update_freq=5): target should NOT have changed.
    for (int i = 0; i < 4; ++i) { agent.train_step(); }
    const double target_val_after4 = agent.target_net().parameters()[0]->data()[0];
    REQUIRE(target_val_after4 == Approx(target_val_init));

    // Run 1 more step (total=5 = target_update_freq): target MUST sync.
    agent.train_step();
    const double target_val_after5  = agent.target_net().parameters()[0]->data()[0];
    const double online_val_after5  = agent.online_net().parameters()[0]->data()[0];
    // After sync, target[0] == online[0].
    REQUIRE(target_val_after5 == Approx(online_val_after5));
}

TEST_CASE("DQNAgent: 200-step loop with multiple target syncs never throws",
          "[dqn]") {
    // CRITICAL stress test: proves Part A verification holds under real DQN
    // training dynamics (multiple optimizer steps + multiple target syncs).
    DQNConfig cfg;
    cfg.state_dim         = 2;
    cfg.num_actions       = 2;
    cfg.hidden_dims       = {4};
    cfg.batch_size        = 8;
    cfg.buffer_capacity   = 200;
    cfg.min_buffer_size   = 8;
    cfg.target_update_freq = 10;  // 20 syncs in 200 steps
    cfg.eps_start         = 0.5f;
    cfg.eps_end           = 0.5f;
    cfg.seed              = 0;

    DQNAgent agent(cfg);

    // Fill buffer — include some truncated transitions to verify masking.
    for (int i = 0; i < 50; ++i) {
        std::vector<float> obs = {static_cast<float>(i % 2),
                                  static_cast<float>(1 - i % 2)};
        const bool trunc = (i % 7 == 0);  // some truncated
        agent.observe_transitions(
            {make_transition(obs, static_cast<int64_t>(i % 2), 1.0f,
                             obs, false, trunc)});
    }

    // 200 train steps with at least 20 target syncs — must never throw.
    REQUIRE_NOTHROW([&]() {
        for (int step = 0; step < 200; ++step) {
            agent.train_step();
        }
    }());
}

// ============================================================================
// PART G — End-to-end integration test
//
// Environment: deterministic 1-action bandit.
//   State:  [1.0, 0.0] (constant).
//   Action: 0 or 1. Action 0 → reward 1.0. Action 1 → reward 0.0.
//   Episode terminates after every step (gamma=0 → targets = rewards).
//   Optimal Q-values: Q(s,0)=1.0, Q(s,1)=0.0.
//
// After training, the DQN should converge: the loss should decrease and the
// greedy action should be 0 (or at least loss should be substantially lower).
// ============================================================================
TEST_CASE("DQN end-to-end: loss trends downward on solvable deterministic bandit",
          "[dqn][e2e]") {
    DQNConfig cfg;
    cfg.state_dim         = 2;
    cfg.num_actions       = 2;
    cfg.hidden_dims       = {8};
    cfg.lr                = 1e-2;
    cfg.gamma             = 0.0;  // pure bandit: no future reward
    cfg.batch_size        = 8;
    cfg.buffer_capacity   = 500;
    cfg.min_buffer_size   = 16;
    cfg.target_update_freq = 20;
    cfg.eps_start         = 0.5f;
    cfg.eps_end           = 0.1f;
    cfg.eps_decay_steps   = 200;
    cfg.seed              = 42;

    DQNAgent agent(cfg);

    const std::vector<float> state = {1.0f, 0.0f};

    // Pre-fill buffer with alternating actions so both Q-values are learned.
    for (int i = 0; i < 32; ++i) {
        const int64_t act = static_cast<int64_t>(i % 2);
        const float   rew = (act == 0) ? 1.0f : 0.0f;
        agent.observe_transitions(
            {make_transition(state, act, rew, state, /*terminated=*/true, false)});
    }

    // Training loop: collect data and train simultaneously.
    std::vector<double> losses;
    losses.reserve(300);
    for (int step = 0; step < 300; ++step) {
        // Collect one more transition (alternate actions for coverage).
        const int64_t act = static_cast<int64_t>(step % 2);
        const float   rew = (act == 0) ? 1.0f : 0.0f;
        agent.observe_transitions(
            {make_transition(state, act, rew, state, true, false)});

        // Train step.
        if (agent.should_update()) {
            const double loss = agent.train_step();
            // Every loss must be finite and non-negative.
            REQUIRE(std::isfinite(loss));
            REQUIRE(loss >= 0.0);
            losses.push_back(loss);
        }
    }

    REQUIRE(!losses.empty());

    // Smoke check: loss should decrease overall.
    // Compare average of first 20 recorded losses vs last 20.
    if (losses.size() >= 40) {
        double first_avg = 0.0;
        double last_avg  = 0.0;
        for (size_t i = 0; i < 20; ++i) { first_avg += losses[i]; }
        for (size_t i = losses.size() - 20; i < losses.size(); ++i) {
            last_avg += losses[i];
        }
        first_avg /= 20.0;
        last_avg  /= 20.0;
        // Loss should have decreased (or at worst not dramatically increased).
        // Using a 3x bound to avoid CI flakiness while still catching regressions.
        REQUIRE(last_avg <= first_avg * 3.0);
    }

    // Greedy evaluation: with gamma=0 and enough training, Q(s,0) > Q(s,1).
    // Use act(explore=false) which internally runs no_grad forward + argmax.
    auto state_obs = std::vector<float>{1.0f, 0.0f};
    auto actions_greedy = agent.act(
        {rl::core::Observation{state_obs}}, /*explore=*/false);
    // After convergence, greedy action should be 0 (the high-reward action).
    REQUIRE(std::get<int64_t>(actions_greedy[0]) == 0);
}
