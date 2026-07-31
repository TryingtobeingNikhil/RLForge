#include "vector_environment_contract.hpp"

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <stdexcept>

#include "rl/core/types.hpp"

namespace rl::testing {

using rl::core::Action;

void run_vector_environment_contract_tests(
    const std::function<std::unique_ptr<rl::core::VectorEnvironment>()>& factory) {

    SECTION("step() before reset() throws") {
        auto vec_env = factory();
        std::mt19937 rng(0);
        std::vector<Action> actions;
        for (size_t i = 0; i < vec_env->num_envs(); ++i) {
            actions.push_back(vec_env->action_space().sample(rng));
        }
        REQUIRE_THROWS_AS(vec_env->step(actions), std::logic_error);
    }

    SECTION("reset() returns num_envs() observations, all within observation_space()") {
        auto vec_env = factory();
        auto result = vec_env->reset(/*seed=*/42);
        REQUIRE(result.observations.size() == vec_env->num_envs());
        REQUIRE(result.infos.size() == vec_env->num_envs());
        for (const auto& obs : result.observations) {
            REQUIRE(vec_env->observation_space().contains(obs));
        }
    }

    SECTION("step() with a mismatched action count throws") {
        auto vec_env = factory();
        vec_env->reset(/*seed=*/42);
        std::mt19937 rng(0);
        std::vector<Action> too_few_actions;
        // Deliberately supply one fewer action than num_envs() requires.
        for (size_t i = 0; i + 1 < vec_env->num_envs(); ++i) {
            too_few_actions.push_back(vec_env->action_space().sample(rng));
        }
        REQUIRE_THROWS_AS(vec_env->step(too_few_actions), std::invalid_argument);
    }

    SECTION("step() returns arrays sized num_envs(), each observation within its space") {
        auto vec_env = factory();
        vec_env->reset(/*seed=*/42);
        std::mt19937 rng(0);
        std::vector<Action> actions;
        for (size_t i = 0; i < vec_env->num_envs(); ++i) {
            actions.push_back(vec_env->action_space().sample(rng));
        }
        auto result = vec_env->step(actions);

        const size_t n = vec_env->num_envs();
        REQUIRE(result.observations.size() == n);
        REQUIRE(result.rewards.size() == n);
        REQUIRE(result.terminated.size() == n);
        REQUIRE(result.truncated.size() == n);
        REQUIRE(result.infos.size() == n);
        REQUIRE(result.final_observations.size() == n);
        for (const auto& obs : result.observations) {
            REQUIRE(vec_env->observation_space().contains(obs));
        }
    }

    SECTION("final_observations is populated iff the corresponding sub-env ended this step") {
        auto vec_env = factory();
        vec_env->reset(/*seed=*/7);
        std::mt19937 rng(7);
        for (int step = 0; step < 200; ++step) {
            std::vector<Action> actions;
            for (size_t i = 0; i < vec_env->num_envs(); ++i) {
                actions.push_back(vec_env->action_space().sample(rng));
            }
            auto result = vec_env->step(actions);
            for (size_t i = 0; i < vec_env->num_envs(); ++i) {
                const bool ended = result.terminated[i] || result.truncated[i];
                REQUIRE(result.final_observations[i].has_value() == ended);
            }
        }
    }

    SECTION("same seed produces the same batched trajectory") {
        auto vec_env_a = factory();
        auto vec_env_b = factory();
        REQUIRE(vec_env_a->num_envs() == vec_env_b->num_envs());

        auto reset_a = vec_env_a->reset(/*seed=*/123);
        auto reset_b = vec_env_b->reset(/*seed=*/123);
        REQUIRE(reset_a.observations == reset_b.observations);

        std::mt19937 action_rng(999);
        for (int step = 0; step < 20; ++step) {
            std::vector<Action> actions;
            for (size_t i = 0; i < vec_env_a->num_envs(); ++i) {
                actions.push_back(vec_env_a->action_space().sample(action_rng));
            }
            auto result_a = vec_env_a->step(actions);
            auto result_b = vec_env_b->step(actions);

            REQUIRE(result_a.observations == result_b.observations);
            REQUIRE(result_a.rewards == result_b.rewards);
            REQUIRE(result_a.terminated == result_b.terminated);
            REQUIRE(result_a.truncated == result_b.truncated);
            REQUIRE(result_a.final_observations == result_b.final_observations);
        }
    }
}

} // namespace rl::testing
