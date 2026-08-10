#pragma once

#include <algorithm>
#include <cstddef>
#include <random>

#include "rl/tensor/tensor.hpp"

namespace rl::agents {

// ---------------------------------------------------------------------------
// EpsilonGreedyPolicy — epsilon-greedy action selection with linear decay.
//
// Epsilon decays linearly from eps_start to eps_end over eps_decay_steps steps.
// After eps_decay_steps steps have been taken, epsilon stays fixed at eps_end.
//
// select_action():
//   With probability epsilon:       select a uniformly random action.
//   With probability (1-epsilon):   select the greedy argmax action.
//
// The RNG is passed in by the caller rather than owned here so that an entire
// training run can be driven from one seeded RNG for reproducibility — the same
// reason ReplayBuffer::sample() takes an rng by reference (rl/core/replay_buffer.hpp).
//
// Argmax tie-breaking: first occurrence (lowest index) wins, consistent with
// QNetwork::forward and Tensor::max_last_dim.
// ---------------------------------------------------------------------------
class EpsilonGreedyPolicy {
public:
    EpsilonGreedyPolicy(float eps_start, float eps_end, size_t eps_decay_steps) noexcept
        : eps_start_(eps_start),
          eps_end_(eps_end),
          eps_decay_steps_(eps_decay_steps),
          step_count_(0) {}

    // Returns the current epsilon value (before incrementing the counter).
    float current_epsilon() const noexcept {
        if (eps_decay_steps_ == 0) { return eps_end_; }
        const float t =
            static_cast<float>(std::min(step_count_, eps_decay_steps_)) /
            static_cast<float>(eps_decay_steps_);
        return eps_start_ + t * (eps_end_ - eps_start_);
    }

    // Select an action given a 1-D Q-value tensor of shape [num_actions].
    // Increments the internal step counter (advancing the decay schedule).
    //
    // q_values must be 1-D with numel() == num_actions.
    // num_actions must be > 0.
    int select_action(const rl::tensor::Tensor& q_values, int num_actions,
                      std::mt19937& rng) {
        const float eps = current_epsilon();
        ++step_count_;

        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        if (coin(rng) < eps) {
            // Random exploration.
            std::uniform_int_distribution<int> act_dist(0, num_actions - 1);
            return act_dist(rng);
        }

        // Greedy: argmax of q_values, first occurrence on tie.
        int best_action = 0;
        double best_val = q_values[static_cast<int64_t>(0)];
        for (int a = 1; a < num_actions; ++a) {
            const double val = q_values[static_cast<int64_t>(a)];
            if (val > best_val) {
                best_val = val;
                best_action = a;
            }
        }
        return best_action;
    }

    size_t step_count()  const noexcept { return step_count_;  }
    float  eps_start()   const noexcept { return eps_start_;   }
    float  eps_end()     const noexcept { return eps_end_;     }
    size_t eps_decay_steps() const noexcept { return eps_decay_steps_; }

private:
    float  eps_start_;
    float  eps_end_;
    size_t eps_decay_steps_;
    size_t step_count_;
};

}  // namespace rl::agents
