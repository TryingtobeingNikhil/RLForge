#include <catch2/catch_test_macros.hpp>

#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

#include "contract/vector_environment_contract.hpp"
#include "rl/envs/grid_world.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/vector_envs/threaded_vector_environment.hpp"

namespace {

std::vector<rl::vector_envs::EnvFactory> factories(size_t count,
                                                    rl::envs::GridWorld::Config config) {
    std::vector<rl::vector_envs::EnvFactory> result;
    for (size_t i = 0; i < count; ++i) {
        result.push_back([config] {
            return std::make_unique<rl::envs::GridWorld>(config);
        });
    }
    return result;
}

} // namespace

TEST_CASE("ThreadedVectorEnvironment satisfies the shared contract",
          "[threaded_vector_env][contract]") {
    rl::testing::run_vector_environment_contract_tests([] {
        rl::envs::GridWorld::Config config;
        config.slip_probability = 0.1f;
        return std::make_unique<rl::vector_envs::ThreadedVectorEnvironment>(
            factories(3, config));
    });
}

TEST_CASE("ThreadedVectorEnvironment preserves terminal observations",
          "[threaded_vector_env]") {
    rl::envs::GridWorld::Config config;
    config.size = 2;
    config.slip_probability = 0.0f;
    rl::vector_envs::ThreadedVectorEnvironment environment(factories(2, config));
    environment.reset(0);

    const rl::core::Action right{int64_t{3}};
    const rl::core::Action up{int64_t{0}};
    environment.step({right, right});
    auto result = environment.step({up, up});
    for (size_t i = 0; i < 2; ++i) {
        REQUIRE(result.terminated[i]);
        REQUIRE(result.final_observations[i].has_value());
        REQUIRE(*result.final_observations[i] ==
                rl::core::Observation{std::vector<float>{1.0f, 1.0f}});
        REQUIRE(result.observations[i] ==
                rl::core::Observation{std::vector<float>{0.0f, 0.0f}});
    }
}

TEST_CASE("ThreadedVectorEnvironment propagates factory failures",
          "[threaded_vector_env]") {
    std::vector<rl::vector_envs::EnvFactory> bad_factories;
    bad_factories.push_back([]() -> std::unique_ptr<rl::core::Environment> {
        throw std::runtime_error("factory failure");
    });
    REQUIRE_THROWS_WITH(
        rl::vector_envs::ThreadedVectorEnvironment(std::move(bad_factories)),
        "factory failure");
}

TEST_CASE("Gradient mode is isolated between threads", "[threaded_vector_env][autograd]") {
    auto guard = rl::tensor::no_grad();
    REQUIRE_FALSE(rl::tensor::grad_mode_enabled());
    auto other_thread = std::async(std::launch::async, [] {
        return rl::tensor::grad_mode_enabled();
    });
    REQUIRE(other_thread.get());
}
