#include "rl/core/transition.hpp"

#include <stdexcept>
#include <utility>

namespace rl::core {

Transition make_transition(Observation observation, Action action,
                            const StepResult& step_result) {
    Transition transition;
    transition.observation = std::move(observation);
    transition.action = std::move(action);
    transition.reward = step_result.reward;
    transition.next_observation = step_result.observation;
    transition.terminated = step_result.terminated;
    transition.truncated = step_result.truncated;
    return transition;
}

std::vector<Transition> make_transitions(
    const std::vector<Observation>& observations,
    const std::vector<Action>& actions, const VectorStepResult& step_result) {
    const size_t n = step_result.observations.size();
    if (observations.size() != n || actions.size() != n) {
        throw std::invalid_argument(
            "make_transitions(): observations.size() and actions.size() "
            "must match step_result's sub-env count (" +
            std::to_string(n) + "); got observations.size()=" +
            std::to_string(observations.size()) +
            " actions.size()=" + std::to_string(actions.size()));
    }

    std::vector<Transition> transitions;
    transitions.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Transition transition;
        transition.observation = observations[i];
        transition.action = actions[i];
        transition.reward = step_result.rewards[i];
        // The critical bit: prefer the true terminal observation over the
        // post-auto-reset one whenever this sub-env's episode just ended.
        transition.next_observation =
            step_result.final_observations[i].has_value()
                ? *step_result.final_observations[i]
                : step_result.observations[i];
        transition.terminated = step_result.terminated[i] != 0;
        transition.truncated = step_result.truncated[i] != 0;
        transitions.push_back(std::move(transition));
    }
    return transitions;
}

} // namespace rl::core
