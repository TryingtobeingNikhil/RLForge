#include "rl/agents/tabular_q_learning_agent.hpp"

#include <cmath>
#include <stdexcept>

namespace rl::agents {

namespace {

// Hashes an Observation's actual contents (not its identity) -- either the
// discrete index, or every float in the continuous vector, combined with
// the standard boost::hash_combine mixing formula. Used only to pick a
// bucket; StateActionKey::operator== (comparing full Observations) is what
// actually decides whether two keys refer to the same state, so a
// collision here can never cause two different states to silently share a
// Q-value.
size_t hash_observation(const rl::core::Observation& observation) {
    if (const auto* index = std::get_if<int64_t>(&observation)) {
        return std::hash<int64_t>{}(*index);
    }
    const auto& values = std::get<std::vector<float>>(observation);
    size_t seed = values.size();
    for (float value : values) {
        seed ^= std::hash<float>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

} // namespace

size_t TabularQLearningAgent::StateActionKeyHash::operator()(const StateActionKey& key) const {
    size_t seed = hash_observation(key.observation);
    seed ^= std::hash<int64_t>{}(key.action) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

TabularQLearningAgent::TabularQLearningAgent(const rl::core::Space& action_space,
                                              TabularQLearningConfig config)
    : config_(config), rng_(config.seed) {
    const auto* discrete = dynamic_cast<const rl::core::Discrete*>(&action_space);
    if (discrete == nullptr) {
        throw std::invalid_argument(
            "TabularQLearningAgent requires a Discrete action space (tabular "
            "Q-Learning is only defined over a finite action set); got " +
            action_space.describe());
    }
    num_actions_ = discrete->n();
}

std::pair<int64_t, float> TabularQLearningAgent::best_action_and_value(
    const rl::core::Observation& observation) const {
    int64_t best_action = 0;
    float best_value = q_value(observation, 0);
    for (int64_t action = 1; action < num_actions_; ++action) {
        const float value = q_value(observation, action);
        if (value > best_value) {
            best_value = value;
            best_action = action;
        }
    }
    return {best_action, best_value};
}

float TabularQLearningAgent::q_value(const rl::core::Observation& observation,
                                      int64_t action) const {
    const auto it = q_table_.find(StateActionKey{observation, action});
    // Unvisited (observation, action) pairs default to 0.0 -- a common,
    // simple choice; this is not optimistic initialization (which would
    // bias early exploration toward untried actions) but it is adequate
    // for GridWorld's bounded, mostly-negative reward structure.
    return it != q_table_.end() ? it->second : 0.0f;
}

float TabularQLearningAgent::current_epsilon() const {
    if (step_count_ >= config_.epsilon_decay_steps) {
        return config_.epsilon_end;
    }
    const float fraction =
        static_cast<float>(step_count_) / static_cast<float>(config_.epsilon_decay_steps);
    return config_.epsilon_start + fraction * (config_.epsilon_end - config_.epsilon_start);
}

std::vector<rl::core::Action> TabularQLearningAgent::act(
    const std::vector<rl::core::Observation>& observations, bool explore) {
    std::vector<rl::core::Action> actions;
    actions.reserve(observations.size());

    std::uniform_real_distribution<float> unit_interval(0.0f, 1.0f);
    std::uniform_int_distribution<int64_t> random_action(0, num_actions_ - 1);

    for (const auto& observation : observations) {
        if (explore && unit_interval(rng_) < current_epsilon()) {
            actions.push_back(rl::core::Action{random_action(rng_)});
        } else {
            actions.push_back(rl::core::Action{best_action_and_value(observation).first});
        }
    }
    return actions;
}

void TabularQLearningAgent::observe_transitions(
    const std::vector<rl::core::Transition>& transitions) {
    pending_transitions_.insert(pending_transitions_.end(), transitions.begin(),
                                 transitions.end());
    step_count_ += transitions.size();
}

rl::core::Metrics TabularQLearningAgent::update() {
    float total_absolute_td_error = 0.0f;

    for (const auto& transition : pending_transitions_) {
        const int64_t action = rl::core::as_index(transition.action);
        const float current_value = q_value(transition.observation, action);

        // The one line this whole milestone's terminated/truncated split
        // exists for: bootstrap through the next state's value UNLESS this
        // transition terminated the true MDP. A truncated (time-limit-cut)
        // transition still bootstraps, because the underlying MDP did not
        // actually end.
        const float next_value = transition.terminated
                                      ? 0.0f
                                      : best_action_and_value(transition.next_observation).second;
        const float td_target = transition.reward + config_.discount_factor * next_value;
        const float td_error = td_target - current_value;

        q_table_[StateActionKey{transition.observation, action}] =
            current_value + config_.learning_rate * td_error;
        total_absolute_td_error += std::abs(td_error);
    }

    const size_t num_updates = pending_transitions_.size();
    pending_transitions_.clear();

    rl::core::Metrics metrics;
    metrics["mean_abs_td_error"] = static_cast<double>(
        num_updates > 0 ? total_absolute_td_error / static_cast<float>(num_updates) : 0.0f);
    metrics["epsilon"] = static_cast<double>(current_epsilon());
    metrics["q_table_size"] = static_cast<int64_t>(q_table_.size());
    return metrics;
}

} // namespace rl::agents
