#include "environment_contract.hpp"

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

#include "rl/core/types.hpp"

namespace rl::testing {

using rl::core::Action;

void run_environment_contract_tests(
    const std::function<std::unique_ptr<rl::core::Environment>()>& factory) {

    SECTION("step() before reset() throws") {
        auto env = factory();
        std::mt19937 rng(0);
        const Action arbitrary_action = env->action_space().sample(rng);
        REQUIRE_THROWS_AS(env->step(arbitrary_action), std::logic_error);
    }

    SECTION("reset() returns an observation within observation_space()") {
        auto env = factory();
        auto result = env->reset(/*seed=*/42);
        REQUIRE(env->observation_space().contains(result.observation));
    }

    SECTION("step() after reset() returns an observation within observation_space()") {
        auto env = factory();
        env->reset(/*seed=*/42);
        std::mt19937 rng(0);
        const Action action = env->action_space().sample(rng);
        auto result = env->step(action);
        REQUIRE(env->observation_space().contains(result.observation));
    }

    SECTION("terminated and truncated are never both true on the same step") {
        // A step that ends the true MDP (terminated) is a different reason
        // for ending an episode than a time limit (truncated). A correct
        // environment should not report both simultaneously -- if it hit a
        // real terminal state, the time limit is irrelevant on that step.
        auto env = factory();
        env->reset(/*seed=*/7);
        std::mt19937 rng(7);
        for (int i = 0; i < 500; ++i) {
            const Action action = env->action_space().sample(rng);
            auto result = env->step(action);
            REQUIRE_FALSE((result.terminated && result.truncated));
            if (result.terminated || result.truncated) {
                env->reset(/*seed=*/static_cast<uint64_t>(i));
            }
        }
    }

    SECTION("same seed produces the same trajectory") {
        auto env_a = factory();
        auto env_b = factory();

        auto reset_a = env_a->reset(/*seed=*/123);
        auto reset_b = env_b->reset(/*seed=*/123);
        REQUIRE(reset_a.observation == reset_b.observation);

        // Drive both environments with the same fixed action sequence
        // (derived from a separately-seeded RNG so the action sequence
        // itself is deterministic across the two runs).
        std::mt19937 action_rng(999);
        for (int i = 0; i < 20; ++i) {
            const Action action = env_a->action_space().sample(action_rng);
            auto result_a = env_a->step(action);
            auto result_b = env_b->step(action);

            REQUIRE(result_a.observation == result_b.observation);
            REQUIRE(result_a.reward == result_b.reward);
            REQUIRE(result_a.terminated == result_b.terminated);
            REQUIRE(result_a.truncated == result_b.truncated);

            if (result_a.terminated || result_a.truncated) {
                break;
            }
        }
    }
}

} // namespace rl::testing
