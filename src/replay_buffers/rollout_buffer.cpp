#include "rl/replay_buffers/rollout_buffer.hpp"

#include <stdexcept>
#include <utility>

namespace rl::replay_buffers {

RolloutBuffer::RolloutBuffer(size_t horizon) : horizon_(horizon) {
    if (horizon_ == 0) {
        throw std::invalid_argument("RolloutBuffer horizon must be positive");
    }
    steps_.reserve(horizon_);
}

void RolloutBuffer::append(RolloutStep step) {
    if (ready()) {
        throw std::logic_error("RolloutBuffer is full; update and clear it first");
    }
    const size_t n = step.rewards.size();
    if (n == 0 || step.observations.size() != n || step.actions.size() != n ||
        step.next_observations.size() != n || step.terminated.size() != n ||
        step.truncated.size() != n || step.log_probabilities.size() != n ||
        step.values.size() != n) {
        throw std::invalid_argument(
            "RolloutStep fields must be non-empty and have identical sizes");
    }
    if (num_envs_ == 0) {
        num_envs_ = n;
    } else if (num_envs_ != n) {
        throw std::invalid_argument("RolloutBuffer vector-env width changed mid-rollout");
    }
    steps_.push_back(std::move(step));
}

void RolloutBuffer::clear() noexcept {
    steps_.clear();
    num_envs_ = 0;
}

} // namespace rl::replay_buffers
