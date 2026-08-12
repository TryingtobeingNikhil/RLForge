#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "rl/agents/ppo.hpp"
#include "rl/core/transition.hpp"

using Catch::Approx;

TEST_CASE("PPO GAE distinguishes termination from truncation", "[ppo][gae]") {
    const std::vector<double> rewards{1.0, 1.0};
    const std::vector<double> values{0.5, 0.5};
    const std::vector<double> next_values{10.0, 10.0};
    const rl::core::BoolArray terminated{1, 0};
    const rl::core::BoolArray truncated{0, 1};

    auto result = rl::agents::compute_gae(rewards, values, next_values,
                                           terminated, truncated,
                                           /*time_steps=*/1, /*num_envs=*/2,
                                           /*gamma=*/0.9, /*lambda=*/0.95);
    REQUIRE(result.advantages[0] == Approx(0.5));
    REQUIRE(result.advantages[1] == Approx(9.5));
    REQUIRE(result.returns[0] == Approx(1.0));
    REQUIRE(result.returns[1] == Approx(10.0));
}

TEST_CASE("PPO GAE keeps vector environment lanes independent", "[ppo][gae]") {
    // [T=2,N=2]. Lane zero terminates at t=0; its t=0 advantage must not
    // include t=1. Lane one remains continuous and does include it.
    const std::vector<double> rewards{1.0, 1.0, 100.0, 2.0};
    const std::vector<double> values(4, 0.0);
    const std::vector<double> next_values(4, 0.0);
    const rl::core::BoolArray terminated{1, 0, 0, 0};
    const rl::core::BoolArray truncated(4, 0);
    auto result = rl::agents::compute_gae(rewards, values, next_values,
                                           terminated, truncated, 2, 2, 1.0, 1.0);
    REQUIRE(result.advantages[0] == Approx(1.0));
    REQUIRE(result.advantages[1] == Approx(3.0));
    REQUIRE(result.advantages[2] == Approx(100.0));
    REQUIRE(result.advantages[3] == Approx(2.0));
}

TEST_CASE("PPO collects a full on-policy rollout then updates", "[ppo]") {
    rl::agents::PPOConfig config;
    config.state_dim = 2;
    config.num_actions = 2;
    config.hidden_dims = {4};
    config.rollout_steps = 2;
    config.update_epochs = 1;
    config.minibatch_size = 4;
    config.seed = 7;
    rl::agents::PPOAgent agent(config);

    std::vector<rl::core::Observation> observations{
        std::vector<float>{0.0f, 0.0f}, std::vector<float>{1.0f, 0.0f}};
    for (size_t step = 0; step < 2; ++step) {
        auto actions = agent.act(observations, true);
        std::vector<rl::core::Transition> transitions;
        for (size_t i = 0; i < observations.size(); ++i) {
            transitions.push_back(rl::core::Transition{
                observations[i], actions[i], 1.0f,
                rl::core::Observation{std::vector<float>{1.0f, 1.0f}},
                false, false});
        }
        agent.observe_transitions(transitions);
    }

    REQUIRE(agent.should_update());
    auto metrics = agent.update();
    REQUIRE(metrics.contains("policy_loss"));
    REQUIRE(metrics.contains("value_loss"));
    REQUIRE(metrics.contains("entropy"));
    REQUIRE(agent.rollout_size() == 0);
    REQUIRE_FALSE(agent.should_update());
}

TEST_CASE("PPO evaluation acts greedily without consuming rollout state", "[ppo]") {
    rl::agents::PPOConfig config;
    config.state_dim = 2;
    config.num_actions = 3;
    config.hidden_dims = {};
    rl::agents::PPOAgent agent(config);
    const std::vector<rl::core::Observation> observations{
        std::vector<float>{0.0f, 1.0f}};
    auto first = agent.act(observations, false);
    auto second = agent.act(observations, false);
    REQUIRE(first == second);
    REQUIRE(agent.rollout_size() == 0);
}
