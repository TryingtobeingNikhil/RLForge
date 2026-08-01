#include <catch2/catch_test_macros.hpp>

#include "rl/core/transition.hpp"

using rl::core::Action;
using rl::core::Observation;
using rl::core::StepResult;
using rl::core::Transition;
using rl::core::VectorStepResult;
using rl::core::make_transition;
using rl::core::make_transitions;

TEST_CASE("make_transition copies StepResult fields directly (no auto-reset ambiguity)", "[transition]") {
    Observation obs{std::vector<float>{1.0f, 2.0f}};
    Action action{int64_t{3}};

    StepResult step_result;
    step_result.observation = Observation{std::vector<float>{4.0f, 5.0f}};
    step_result.reward = -1.0f;
    step_result.terminated = true;
    step_result.truncated = false;

    Transition transition = make_transition(obs, action, step_result);

    REQUIRE(transition.observation == obs);
    REQUIRE(transition.action == action);
    REQUIRE(transition.reward == -1.0f);
    REQUIRE(transition.next_observation == step_result.observation);
    REQUIRE(transition.terminated);
    REQUIRE_FALSE(transition.truncated);
}

TEST_CASE("make_transitions uses final_observations for ended sub-envs, observations otherwise", "[transition]") {
    // Two sub-envs: sub-env 0 ended this step (terminated), sub-env 1 did not.
    std::vector<Observation> previous_observations{
        Observation{std::vector<float>{0.0f, 0.0f}},
        Observation{std::vector<float>{1.0f, 1.0f}},
    };
    std::vector<Action> actions{Action{int64_t{0}}, Action{int64_t{1}}};

    VectorStepResult step_result;
    step_result.observations = {
        // sub-env 0's post-auto-reset observation -- must NOT end up as
        // next_observation for sub-env 0's transition.
        Observation{std::vector<float>{0.0f, 0.0f}},
        Observation{std::vector<float>{2.0f, 2.0f}},
    };
    step_result.rewards = {10.0f, -1.0f};
    step_result.terminated = {1, 0};
    step_result.truncated = {0, 0};
    step_result.infos = {{}, {}};
    step_result.final_observations = {
        // sub-env 0's TRUE terminal observation.
        std::make_optional(Observation{std::vector<float>{9.0f, 9.0f}}),
        std::nullopt,
    };

    auto transitions = make_transitions(previous_observations, actions, step_result);

    REQUIRE(transitions.size() == 2);

    // Sub-env 0: ended this step -> next_observation must be the terminal
    // observation (9, 9), NOT the post-reset observation (0, 0).
    REQUIRE(transitions[0].terminated);
    REQUIRE(transitions[0].next_observation ==
            Observation{std::vector<float>{9.0f, 9.0f}});
    REQUIRE(transitions[0].reward == 10.0f);

    // Sub-env 1: did not end -> next_observation is just observations[1].
    REQUIRE_FALSE(transitions[1].terminated);
    REQUIRE(transitions[1].next_observation ==
            Observation{std::vector<float>{2.0f, 2.0f}});
    REQUIRE(transitions[1].reward == -1.0f);
}

TEST_CASE("make_transitions rejects mismatched observations/actions sizes", "[transition]") {
    VectorStepResult step_result;
    step_result.observations = {Observation{std::vector<float>{0.0f}},
                                 Observation{std::vector<float>{1.0f}}};
    step_result.rewards = {0.0f, 0.0f};
    step_result.terminated = {0, 0};
    step_result.truncated = {0, 0};
    step_result.infos = {{}, {}};
    step_result.final_observations = {std::nullopt, std::nullopt};

    std::vector<Observation> one_observation{Observation{std::vector<float>{0.0f}}};
    std::vector<Action> two_actions{Action{int64_t{0}}, Action{int64_t{1}}};

    REQUIRE_THROWS_AS(make_transitions(one_observation, two_actions, step_result),
                       std::invalid_argument);
}
