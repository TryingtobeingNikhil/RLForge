#pragma once

#include <cstddef>
#include <vector>

#include "rl/core/transition.hpp"
#include "rl/core/types.hpp"

namespace rl::replay_buffers {

struct RolloutStep {
    std::vector<rl::core::Observation> observations;
    std::vector<rl::core::Action> actions;
    std::vector<float> rewards;
    std::vector<rl::core::Observation> next_observations;
    rl::core::BoolArray terminated;
    rl::core::BoolArray truncated;
    std::vector<double> log_probabilities;
    std::vector<double> values;
};

// Fixed-horizon on-policy storage. A step contains one item per vector-env
// lane, preserving lane identity for reverse-time GAE computation.
class RolloutBuffer {
public:
    explicit RolloutBuffer(size_t horizon);

    void append(RolloutStep step);
    bool ready() const noexcept { return steps_.size() == horizon_; }
    size_t size() const noexcept { return steps_.size(); }
    size_t horizon() const noexcept { return horizon_; }
    size_t num_envs() const noexcept { return num_envs_; }
    const std::vector<RolloutStep>& steps() const noexcept { return steps_; }
    void clear() noexcept;

private:
    size_t horizon_;
    size_t num_envs_ = 0;
    std::vector<RolloutStep> steps_;
};

} // namespace rl::replay_buffers
