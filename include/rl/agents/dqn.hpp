#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rl/agents/epsilon_greedy.hpp"
#include "rl/core/agent.hpp"
#include "rl/core/replay_buffer.hpp"
#include "rl/core/transition.hpp"
#include "rl/core/types.hpp"
#include "rl/nn/qnetwork.hpp"
#include "rl/optim/adam.hpp"
#include "rl/replay_buffers/vector_transition_storage.hpp"

namespace rl::agents {

// ---------------------------------------------------------------------------
// DQNConfig — hyperparameters for DQNAgent.
//
// Declared as a free (non-nested) struct to avoid constructor default-argument
// resolution issues seen with nested struct types on standards-conforming
// compilers (same pattern as TabularQLearningConfig and GridWorldConfig).
// ---------------------------------------------------------------------------
struct DQNConfig {
    int64_t state_dim   = 4;
    int64_t num_actions = 2;
    std::vector<int64_t> hidden_dims = {64, 64};

    double lr    = 1e-3;
    double gamma = 0.99;

    size_t batch_size       = 32;
    size_t buffer_capacity  = 10000;
    size_t min_buffer_size  = 64;    // warmup: don't learn until buffer has this many
    size_t target_update_freq = 100; // hard-sync target every this many train steps

    float  eps_start      = 1.0f;
    float  eps_end        = 0.05f;
    size_t eps_decay_steps = 10000;

    uint64_t seed = 0;
};

// ---------------------------------------------------------------------------
// DQNAgent — vanilla Deep Q-Network (Mnih et al., 2015).
//
// Implements rl::core::Agent so it can be driven transparently by Trainer.
//
// ARCHITECTURE:
//   online_net_ — trainable Q-network (requires_grad=true params).
//   target_net_ — frozen reference Q-network (requires_grad=false params).
//     Parameters are hard-synced from online_net_ every target_update_freq
//     train steps. Forward passes through target_net_ are always done under
//     no_grad() — doubly safe given requires_grad=false on all its parameters.
//
// TRAINING LOOP (train_step()):
//   1. Sample batch from replay buffer.
//   2. Convert to Tensors via rl::data::batch_to_tensors.
//   3. q_values  = gather(online_net.forward(states), actions)       [B]
//   4. next_q_max = max_last_dim(target_net.forward(next_states))    [B]
//      (under no_grad())
//   5. targets  = (rewards + gamma * next_q_max * (1-terminated)).detach()
//   6. loss     = mse_loss(q_values, targets)
//   7. zero_grad; loss.backward(); optimizer.step()
//   8. Increment step counter; hard-sync target if step % freq == 0.
//
// TERMINATED vs TRUNCATED (CRITICAL CORRECTNESS):
//   The bootstrap mask is (1 - terminated). ONLY the `terminated` flag from
//   the Transition struct zeros the bootstrap, NOT `truncated`. A truncated
//   transition was cut off by a time limit — the underlying MDP did NOT end,
//   so the value estimate at next_state is still valid and must be kept.
//   This is the single most common DQN correctness bug.
//
// VERSION-GUARD SAFETY:
//   Target sync bumps only target parameter Storage versions (via data_mutable()).
//   Online training bumps only online parameter Storage versions (via Adam::step()).
//   The two networks have completely separate Storage objects — they can never
//   interfere with each other's version counters or each other's graphs.
// ---------------------------------------------------------------------------
class DQNAgent final : public rl::core::Agent {
public:
    explicit DQNAgent(DQNConfig config = {});

    // act(): for each observation, run a no_grad forward pass through online_net_
    //        and select an action (epsilon-greedy if explore=true, else greedy).
    std::vector<rl::core::Action> act(
        const std::vector<rl::core::Observation>& observations, bool explore) override;

    // observe_transitions(): push each transition into the replay buffer.
    //   Does NOT perform a learning update — pure data intake per Agent contract.
    void observe_transitions(const std::vector<rl::core::Transition>& transitions) override;

    // should_update(): true when replay buffer has >= min_buffer_size transitions.
    bool should_update() const override;

    // update(): perform one train_step() and return {"loss": <double>}.
    rl::core::Metrics update() override;

    std::string name() const override { return "DQN"; }

    // -----------------------------------------------------------------------
    // Direct access for testing and introspection.
    // -----------------------------------------------------------------------

    // Execute one DQN training step and return the loss value.
    // Caller must ensure replay_.size() >= config_.batch_size.
    double train_step();

    const rl::nn::QNetwork& online_net()       const { return online_net_; }
    const rl::nn::QNetwork& target_net()       const { return target_net_; }
    size_t                  train_step_count() const noexcept { return train_step_count_; }
    size_t                  buffer_size()      const noexcept { return replay_.size(); }

private:
    // Copy all online_net_ parameter values into target_net_ via data_mutable().
    // Increments each target parameter's Storage version counter.
    // Does NOT touch online_net_ parameters or their version counters.
    void sync_target_from_online();

    DQNConfig          config_;
    rl::nn::QNetwork   online_net_;
    rl::nn::QNetwork   target_net_;
    rl::optim::Adam    optimizer_;    // over online_net_.parameters() only
    rl::core::ReplayBuffer replay_;
    EpsilonGreedyPolicy    policy_;
    std::mt19937           rng_;
    size_t                 train_step_count_ = 0;
};

}  // namespace rl::agents
