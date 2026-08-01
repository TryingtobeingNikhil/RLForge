#pragma once

#include <cstddef>
#include <vector>

#include "rl/core/replay_buffer.hpp"

namespace rl::replay_buffers {

// The reference TransitionStorage backend: transitions are stored exactly
// as produced by an Environment, in a std::vector<Transition>, each holding
// its Observation/Action as a std::variant<int64_t, std::vector<float>>.
// Simple, and allocation-heavy -- every continuous observation/action owns
// its own heap-allocated vector<float> -- but correct, and adequate until
// the neural network milestone determines what contiguous layout is
// actually worth optimizing for. See rl/core/replay_buffer.hpp for why this
// is a swappable implementation detail rather than baked into ReplayBuffer.
class VectorTransitionStorage final : public rl::core::TransitionStorage {
public:
    explicit VectorTransitionStorage(size_t capacity);

    void set(size_t index, rl::core::Transition transition) override;
    rl::core::Transition get(size_t index) const override;
    size_t capacity() const override { return storage_.size(); }

private:
    std::vector<rl::core::Transition> storage_;
};

} // namespace rl::replay_buffers
