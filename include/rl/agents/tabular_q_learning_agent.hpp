#pragma once

#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rl/core/agent.hpp"
#include "rl/core/space.hpp"

namespace rl::agents {

// Free (non-nested) struct, not TabularQLearningAgent::Config defined
// in-class -- see the comment on GridWorldConfig in
// rl/envs/grid_world.hpp for why a nested type used in a constructor
// default argument (`= {}`) doesn't reliably resolve on standards-
// conforming compilers.
struct TabularQLearningConfig {
    float learning_rate = 0.1f;
    float discount_factor = 0.99f;
    float epsilon_start = 1.0f;
    float epsilon_end = 0.05f;
    size_t epsilon_decay_steps = 10000;
    uint64_t seed = 0;
};

// The reference implementation of rl::core::Agent: classic tabular
// Q-Learning with epsilon-greedy exploration. Exists in this milestone to
// validate the Agent/Trainer abstractions end to end on GridWorld before
// any neural network machinery exists -- not as a permanent production
// algorithm.
//
// Requires a Discrete action space (checked at construction); tabular
// Q-Learning is fundamentally only defined over a finite action set. It
// does NOT require a Discrete *observation* space, though -- the Q-table is
// keyed by (Observation, action) pairs using Observation's own equality
// and a hash over its contents (see the .cpp), so it works directly against
// GridWorld's existing Box(x, y) observation without requiring Milestone 1
// to change how GridWorld represents state.
class TabularQLearningAgent final : public rl::core::Agent {
public:
    explicit TabularQLearningAgent(const rl::core::Space& action_space,
                                    TabularQLearningConfig config = {});

    std::vector<rl::core::Action> act(
        const std::vector<rl::core::Observation>& observations, bool explore) override;
    void observe_transitions(
        const std::vector<rl::core::Transition>& transitions) override;
    bool should_update() const override { return !pending_transitions_.empty(); }
    rl::core::Metrics update() override;
    std::string name() const override { return "TabularQLearning"; }

    // Introspection, useful for debugging/visualization and for tests that
    // verify specific TD-update arithmetic. Not part of the Agent
    // interface -- these are TabularQLearningAgent-specific.
    float q_value(const rl::core::Observation& observation, int64_t action) const;
    int64_t num_actions() const noexcept { return num_actions_; }
    size_t table_size() const noexcept { return q_table_.size(); }

private:
    // Key = (observation, action). Hashing is over the Observation's actual
    // contents (see StateActionKeyHash in the .cpp), with the full
    // Observation stored and compared on lookup -- so a hash collision
    // between two different observations can never silently conflate their
    // Q-values; std::unordered_map's equality check (StateActionKey::
    // operator==) is what actually decides identity.
    struct StateActionKey {
        rl::core::Observation observation;
        int64_t action;
        bool operator==(const StateActionKey& other) const {
            return action == other.action && observation == other.observation;
        }
    };
    struct StateActionKeyHash {
        size_t operator()(const StateActionKey& key) const;
    };

    // Returns both the greedy action and its Q-value in one table scan;
    // used by act() (for the action) and update() (for the next-state
    // value in the TD target) so the two don't duplicate the scan logic.
    std::pair<int64_t, float> best_action_and_value(
        const rl::core::Observation& observation) const;

    float current_epsilon() const;

    TabularQLearningConfig config_;
    int64_t num_actions_;
    std::unordered_map<StateActionKey, float, StateActionKeyHash> q_table_;
    std::vector<rl::core::Transition> pending_transitions_;
    std::mt19937 rng_;
    size_t step_count_ = 0;
};

} // namespace rl::agents
