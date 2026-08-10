#include "rl/agents/dqn.hpp"

#include <cmath>
#include <stdexcept>

#include "rl/core/types.hpp"
#include "rl/data/batch_to_tensors.hpp"
#include "rl/nn/losses.hpp"
#include "rl/tensor/autograd.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::agents {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DQNAgent::DQNAgent(DQNConfig config)
    : config_(std::move(config)),
      online_net_(config_.state_dim, config_.hidden_dims, config_.num_actions),
      target_net_(config_.state_dim, config_.hidden_dims, config_.num_actions),
      optimizer_(online_net_.parameters(), config_.lr),
      replay_(std::make_unique<rl::replay_buffers::VectorTransitionStorage>(
          config_.buffer_capacity)),
      policy_(config_.eps_start, config_.eps_end, config_.eps_decay_steps),
      rng_(config_.seed) {
    // Target network: set all parameters to requires_grad=false.
    // target_net_ is ONLY updated via hard sync (data_mutable()), never via
    // the optimizer. Gradient should not flow through it.
    for (auto& p : target_net_.parameters()) {
        p->requires_grad_(false);
    }
    // Initialise target network with the same weights as online network.
    sync_target_from_online();
}

// ---------------------------------------------------------------------------
// Agent interface
// ---------------------------------------------------------------------------

std::vector<rl::core::Action> DQNAgent::act(
    const std::vector<rl::core::Observation>& observations, bool explore) {
    std::vector<rl::core::Action> actions;
    actions.reserve(observations.size());

    for (const auto& obs : observations) {
        // Convert observation to a [1, state_dim] Tensor.
        const auto& obs_vec = rl::core::as_vector(obs);
        std::vector<double> state_data;
        state_data.reserve(obs_vec.size());
        for (float f : obs_vec) { state_data.push_back(static_cast<double>(f)); }

        auto state = rl::tensor::Tensor::from_data(
            std::move(state_data), {1, config_.state_dim});

        // Forward pass under no_grad() — action selection never needs a graph.
        rl::tensor::Tensor q_row({1, config_.num_actions});
        {
            auto guard = rl::tensor::no_grad();
            q_row = online_net_.forward(state);  // [1, num_actions]
        }

        // Flatten [1, num_actions] -> [num_actions] for the policy.
        auto q_flat = rl::tensor::Tensor::from_data(q_row.data(), {config_.num_actions});

        int action_idx;
        if (explore) {
            action_idx = policy_.select_action(
                q_flat, static_cast<int>(config_.num_actions), rng_);
        } else {
            // Greedy argmax — do not advance the decay counter during evaluation.
            action_idx = 0;
            double best_v = q_flat[static_cast<int64_t>(0)];
            for (int a = 1; a < static_cast<int>(config_.num_actions); ++a) {
                const double v = q_flat[static_cast<int64_t>(a)];
                if (v > best_v) { best_v = v; action_idx = a; }
            }
        }
        actions.push_back(static_cast<int64_t>(action_idx));
    }
    return actions;
}

void DQNAgent::observe_transitions(
    const std::vector<rl::core::Transition>& transitions) {
    // Pure data intake — no learning update here (Agent interface contract).
    for (const auto& t : transitions) {
        replay_.add(t);
    }
}

bool DQNAgent::should_update() const {
    return replay_.size() >= config_.min_buffer_size;
}

rl::core::Metrics DQNAgent::update() {
    const double loss_val = train_step();
    return rl::core::Metrics{{"loss", loss_val}};
}

// ---------------------------------------------------------------------------
// DQN training step
// ---------------------------------------------------------------------------

double DQNAgent::train_step() {
    // Step 1: Sample a batch from the replay buffer.
    auto raw_batch = replay_.sample(config_.batch_size, rng_);

    // Step 2: Convert to Tensors via the algorithm-agnostic batch_to_tensors.
    auto tb = rl::data::batch_to_tensors(raw_batch, config_.state_dim);
    // tb.states       [B, state_dim]  — no requires_grad
    // tb.actions      [B]             — int indices as double, for gather()
    // tb.rewards      [B]             — float rewards
    // tb.next_states  [B, state_dim]  — no requires_grad
    // tb.terminated   [B]             — 1.0=terminated, 0.0=not (truncated excluded)

    // Step 3: Compute current Q-values via ONLINE network (graph IS built here).
    auto q_all    = online_net_.forward(tb.states);   // [B, num_actions], rg=true
    auto q_values = q_all.gather(tb.actions);         // [B], rg=true

    // Step 4: Compute max next-Q via TARGET network under no_grad().
    //   target params have rg=false AND no_grad() is active — doubly safe.
    rl::tensor::Tensor next_q_max =
        rl::tensor::Tensor::zeros({static_cast<int64_t>(config_.batch_size)});
    {
        auto guard = rl::tensor::no_grad();
        auto next_q_all = target_net_.forward(tb.next_states); // [B, num_actions]
        next_q_max = next_q_all.max_last_dim();                // [B]
    }
    // next_q_max has no graph (no_grad + rg=false on target params).

    // Step 5: Compute Bellman targets.
    //   y_i = r_i + gamma * max_a Q_target(s'_i, a) * (1 - terminated_i)
    //
    // CORRECTNESS: (1 - terminated) is the bootstrap mask.
    //   terminated_i=1 → (1-1)=0 → no bootstrap (MDP truly ended).
    //   truncated_i=1  → terminated_i=0 → (1-0)=1 → bootstrap kept.
    //   This is correct: truncation is a time-limit artefact, not an MDP end.
    const int64_t B = static_cast<int64_t>(config_.batch_size);
    auto one_minus_term = rl::tensor::Tensor::ones({B}).sub(tb.terminated); // [B]
    auto bootstrap      = next_q_max.mul(one_minus_term).mul(config_.gamma); // [B]
    auto targets        = tb.rewards.add(bootstrap).detach();               // [B], no graph

    // Step 6: MSE loss between predicted Q-values and Bellman targets.
    auto loss = rl::nn::mse_loss(q_values, targets);  // scalar

    // Step 7: Gradient update on ONLINE network only.
    optimizer_.zero_grad();
    loss.backward();
    optimizer_.step();
    // After step(), all online parameter Storage versions have been bumped.
    // The graph from this iteration is stale — but we never call backward()
    // on it again, so the version guard never fires during normal operation.

    // Step 8: Increment train step counter; hard-sync target if scheduled.
    ++train_step_count_;
    if (config_.target_update_freq > 0 &&
        train_step_count_ % config_.target_update_freq == 0) {
        sync_target_from_online();
    }

    return loss.item();
}

// ---------------------------------------------------------------------------
// Target network synchronisation
// ---------------------------------------------------------------------------

void DQNAgent::sync_target_from_online() {
    // Copy each online parameter's data into the corresponding target parameter
    // via data_mutable() (which bumps the target Storage version).
    // Online Storage versions are NEVER touched here — the two networks are
    // completely independent, with separate Storage objects.
    auto online_params = online_net_.parameters();
    auto target_params = target_net_.parameters();
    for (size_t i = 0; i < online_params.size(); ++i) {
        auto& dst = target_params[i]->data_mutable();  // bumps target version
        dst = online_params[i]->data();
    }
}

}  // namespace rl::agents
