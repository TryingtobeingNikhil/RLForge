#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "rl/core/agent.hpp"
#include "rl/nn/actor_critic.hpp"
#include "rl/optim/adam.hpp"
#include "rl/replay_buffers/rollout_buffer.hpp"

namespace rl::agents {

struct PPOConfig {
    int64_t state_dim = 4;
    int64_t num_actions = 2;
    std::vector<int64_t> hidden_dims = {64, 64};

    double learning_rate = 3e-4;
    double gamma = 0.99;
    double gae_lambda = 0.95;
    double clip_range = 0.2;
    double value_loss_coefficient = 0.5;
    double entropy_coefficient = 0.01;

    size_t rollout_steps = 128;
    size_t update_epochs = 4;
    size_t minibatch_size = 64;
    bool normalize_advantages = true;
    uint64_t seed = 0;
};

struct GAEOutput {
    std::vector<double> advantages;
    std::vector<double> returns;
};

// Computes reverse-time generalized advantage estimates for flattened [T,N]
// arrays. Termination disables bootstrapping; both termination and truncation
// break the recursive trace so advantages never cross episode boundaries.
GAEOutput compute_gae(const std::vector<double>& rewards,
                      const std::vector<double>& values,
                      const std::vector<double>& next_values,
                      const rl::core::BoolArray& terminated,
                      const rl::core::BoolArray& truncated,
                      size_t time_steps, size_t num_envs,
                      double gamma, double gae_lambda);

class PPOAgent final : public rl::core::Agent {
public:
    explicit PPOAgent(PPOConfig config = {});

    std::vector<rl::core::Action> act(
        const std::vector<rl::core::Observation>& observations,
        bool explore) override;
    void observe_transitions(
        const std::vector<rl::core::Transition>& transitions) override;
    bool should_update() const override;
    rl::core::Metrics update() override;
    std::string name() const override { return "PPO"; }

    size_t rollout_size() const noexcept { return rollout_.size(); }
    const rl::nn::ActorCriticNetwork& network() const noexcept { return network_; }

private:
    struct PendingActionBatch {
        std::vector<rl::core::Action> actions;
        std::vector<double> log_probabilities;
        std::vector<double> values;
    };

    static rl::tensor::Tensor observations_to_tensor(
        const std::vector<rl::core::Observation>& observations,
        int64_t state_dim);

    PPOConfig config_;
    rl::nn::ActorCriticNetwork network_;
    rl::optim::Adam optimizer_;
    rl::replay_buffers::RolloutBuffer rollout_;
    std::mt19937 rng_;
    std::optional<PendingActionBatch> pending_actions_;
};

} // namespace rl::agents
