#include <catch2/catch_test_macros.hpp>

#include "contract/environment_contract.hpp"
#include "rl/core/types.hpp"
#include "rl/envs/grid_world.hpp"

using rl::envs::GridWorld;

TEST_CASE("GridWorld satisfies the Environment contract", "[grid_world][contract]") {
    rl::testing::run_environment_contract_tests(
        []() -> std::unique_ptr<rl::core::Environment> {
            GridWorld::Config config;
            config.size = 5;
            config.max_episode_steps = 100;
            config.slip_probability = 0.1f;
            return std::make_unique<GridWorld>(config);
        });
}

TEST_CASE("GridWorld reaching the goal terminates with positive reward", "[grid_world]") {
    GridWorld::Config config;
    config.size = 3;
    config.slip_probability = 0.0f; // deterministic for this test
    GridWorld env(config);
    env.reset(/*seed=*/1);

    // 3x3 grid, agent starts at (0,0), goal at (2,2). Right, Right, Up, Up
    // reaches the goal deterministically with slip disabled.
    using rl::core::Action;
    env.step(Action{int64_t{static_cast<int64_t>(GridWorld::GridAction::Right)}});
    auto result = env.step(Action{int64_t{static_cast<int64_t>(GridWorld::GridAction::Right)}});
    REQUIRE_FALSE(result.terminated);

    env.step(Action{int64_t{static_cast<int64_t>(GridWorld::GridAction::Up)}});
    result = env.step(Action{int64_t{static_cast<int64_t>(GridWorld::GridAction::Up)}});

    REQUIRE(result.terminated);
    REQUIRE_FALSE(result.truncated);
    REQUIRE(result.reward == 10.0f);
}

TEST_CASE("GridWorld truncates after max_episode_steps without reaching the goal", "[grid_world]") {
    GridWorld::Config config;
    config.size = 10; // large enough that bouncing off one wall won't reach it
    config.max_episode_steps = 5;
    config.slip_probability = 0.0f;
    GridWorld env(config);
    env.reset(/*seed=*/1);

    using rl::core::Action;
    // Repeatedly move Down; agent is clamped at y=0 and never reaches the
    // goal at (9, 9), so this must truncate rather than terminate.
    rl::core::StepResult result;
    for (int i = 0; i < config.max_episode_steps; ++i) {
        result = env.step(Action{int64_t{static_cast<int64_t>(GridWorld::GridAction::Down)}});
    }

    REQUIRE(result.truncated);
    REQUIRE_FALSE(result.terminated);
}

TEST_CASE("GridWorld rejects an action outside its action_space", "[grid_world]") {
    GridWorld env;
    env.reset(/*seed=*/1);

    using rl::core::Action;
    REQUIRE_THROWS_AS(env.step(Action{int64_t{99}}), std::invalid_argument);
    REQUIRE_THROWS_AS(env.step(Action{std::vector<float>{1.0f, 2.0f}}),
                       std::invalid_argument);
}
