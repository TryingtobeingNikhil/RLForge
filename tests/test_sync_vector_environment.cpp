#include <catch2/catch_test_macros.hpp>

#include "contract/vector_environment_contract.hpp"
#include "rl/core/types.hpp"
#include "rl/envs/grid_world.hpp"
#include "rl/vector_envs/sync_vector_environment.hpp"

using rl::envs::GridWorld;
using rl::vector_envs::EnvFactory;
using rl::vector_envs::SyncVectorEnvironment;

namespace {

// Builds `count` GridWorld factories sharing an identical config, so the
// resulting SyncVectorEnvironment has homogeneous sub-envs (as required).
std::vector<EnvFactory> make_grid_world_factories(size_t count, GridWorld::Config config) {
    std::vector<EnvFactory> factories;
    factories.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        factories.push_back([config]() -> std::unique_ptr<rl::core::Environment> {
            return std::make_unique<GridWorld>(config);
        });
    }
    return factories;
}

} // namespace

TEST_CASE("SyncVectorEnvironment satisfies the VectorEnvironment contract", "[vector_env][contract]") {
    rl::testing::run_vector_environment_contract_tests(
        []() -> std::unique_ptr<rl::core::VectorEnvironment> {
            GridWorld::Config config;
            config.size = 5;
            config.max_episode_steps = 100;
            config.slip_probability = 0.1f;
            return std::make_unique<SyncVectorEnvironment>(
                make_grid_world_factories(3, config));
        });
}

TEST_CASE("SyncVectorEnvironment rejects an empty factory list", "[vector_env]") {
    std::vector<EnvFactory> empty_factories;
    REQUIRE_THROWS_AS(SyncVectorEnvironment(std::move(empty_factories)),
                       std::invalid_argument);
}

TEST_CASE("SyncVectorEnvironment rejects sub-envs with mismatched observation spaces", "[vector_env]") {
    std::vector<EnvFactory> factories;
    GridWorld::Config small_config;
    small_config.size = 3;
    GridWorld::Config large_config;
    large_config.size = 5;

    factories.push_back([small_config]() -> std::unique_ptr<rl::core::Environment> {
        return std::make_unique<GridWorld>(small_config);
    });
    factories.push_back([large_config]() -> std::unique_ptr<rl::core::Environment> {
        return std::make_unique<GridWorld>(large_config);
    });

    REQUIRE_THROWS_AS(SyncVectorEnvironment(std::move(factories)), std::invalid_argument);
}

TEST_CASE("SyncVectorEnvironment auto-resets a sub-env and preserves its terminal observation", "[vector_env]") {
    // A 2x2 grid: agent starts (0,0), goal is (1,1), reachable in exactly
    // two steps with slip disabled, regardless of order (Right then Up).
    GridWorld::Config config;
    config.size = 2;
    config.max_episode_steps = 1000;
    config.slip_probability = 0.0f;

    SyncVectorEnvironment vec_env(make_grid_world_factories(2, config));
    auto reset_result = vec_env.reset(/*seed=*/0);
    for (const auto& obs : reset_result.observations) {
        REQUIRE(obs == rl::core::Observation{std::vector<float>{0.0f, 0.0f}});
    }

    using rl::core::Action;
    const Action right{int64_t{static_cast<int64_t>(GridWorld::GridAction::Right)}};
    const Action up{int64_t{static_cast<int64_t>(GridWorld::GridAction::Up)}};

    // Step 1: move both sub-envs Right -> (1, 0), not yet terminal.
    auto step1 = vec_env.step({right, right});
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE_FALSE(step1.terminated[i]);
        REQUIRE_FALSE(step1.truncated[i]);
        REQUIRE_FALSE(step1.final_observations[i].has_value());
        REQUIRE(step1.observations[i] ==
                rl::core::Observation{std::vector<float>{1.0f, 0.0f}});
    }

    // Step 2: move both sub-envs Up -> reach (1, 1) -> terminated, auto-reset.
    auto step2 = vec_env.step({up, up});
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(step2.terminated[i]);
        REQUIRE_FALSE(step2.truncated[i]);
        REQUIRE(step2.rewards[i] == 10.0f);

        // The true terminal observation is the goal, preserved separately...
        REQUIRE(step2.final_observations[i].has_value());
        REQUIRE(*step2.final_observations[i] ==
                rl::core::Observation{std::vector<float>{1.0f, 1.0f}});

        // ...while the "current" observation is already the post-reset
        // initial state, ready for the next call to step().
        REQUIRE(step2.observations[i] ==
                rl::core::Observation{std::vector<float>{0.0f, 0.0f}});
    }
}
