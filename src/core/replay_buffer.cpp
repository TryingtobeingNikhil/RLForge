#include "rl/core/replay_buffer.hpp"

#include <stdexcept>
#include <utility>

namespace rl::core {

ReplayBuffer::ReplayBuffer(std::unique_ptr<TransitionStorage> storage)
    : storage_(std::move(storage)) {
    if (!storage_) {
        throw std::invalid_argument(
            "ReplayBuffer requires a non-null storage backend");
    }
    capacity_ = storage_->capacity();
    if (capacity_ == 0) {
        throw std::invalid_argument(
            "ReplayBuffer's storage backend must have capacity > 0");
    }
}

void ReplayBuffer::add(Transition transition) {
    storage_->set(next_index_, std::move(transition));
    next_index_ = (next_index_ + 1) % capacity_;
    if (size_ < capacity_) {
        ++size_;
    }
}

TransitionBatch ReplayBuffer::sample(size_t batch_size, std::mt19937& rng) const {
    if (size_ == 0) {
        throw std::logic_error(
            "ReplayBuffer::sample() called on an empty buffer -- add() at "
            "least one transition first.");
    }

    TransitionBatch batch;
    batch.observations.reserve(batch_size);
    batch.actions.reserve(batch_size);
    batch.rewards.reserve(batch_size);
    batch.next_observations.reserve(batch_size);
    batch.terminated.reserve(batch_size);
    batch.truncated.reserve(batch_size);

    std::uniform_int_distribution<size_t> index_dist(0, size_ - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        Transition transition = storage_->get(index_dist(rng));
        batch.observations.push_back(std::move(transition.observation));
        batch.actions.push_back(std::move(transition.action));
        batch.rewards.push_back(transition.reward);
        batch.next_observations.push_back(std::move(transition.next_observation));
        batch.terminated.push_back(transition.terminated ? 1 : 0);
        batch.truncated.push_back(transition.truncated ? 1 : 0);
    }

    return batch;
}

} // namespace rl::core
