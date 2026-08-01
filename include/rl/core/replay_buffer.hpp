#pragma once

#include <cstddef>
#include <memory>
#include <random>

#include "rl/core/transition.hpp"

namespace rl::core {

// Backing storage for a ReplayBuffer's transitions. ReplayBuffer owns the
// policy -- ring-buffer index management and the sampling strategy;
// TransitionStorage owns only how a Transition is physically represented in
// memory. This split is what lets the backend move from the current
// Value-based representation (rl::replay_buffers::VectorTransitionStorage)
// to flat, contiguous float buffers later -- once the neural network
// milestone determines what layout is actually worth optimizing for --
// without ReplayBuffer's public API, or any algorithm code that samples
// from it, changing at all.
class TransitionStorage {
public:
    virtual ~TransitionStorage() = default;

    // Writes `transition` into slot `index`. `index` is always < capacity()
    // -- the caller (ReplayBuffer) owns all ring-buffer index arithmetic; a
    // TransitionStorage implementation never wraps around or evicts on its
    // own initiative.
    virtual void set(size_t index, Transition transition) = 0;

    // Reads back the transition at `index`. `index` is always < capacity()
    // and always an index that has previously been set(); a
    // TransitionStorage need not define behavior for an unwritten index,
    // since ReplayBuffer never reads one.
    virtual Transition get(size_t index) const = 0;

    // Fixed for the lifetime of the storage instance. ReplayBuffer derives
    // its own capacity from this rather than accepting a separate capacity
    // argument, so the two can never disagree.
    virtual size_t capacity() const = 0;
};

// ReplayBuffer is a fixed-capacity ring buffer of Transitions, sampled
// uniformly at random (with replacement) for training. It owns the sampling
// policy and ring-buffer bookkeeping (current size, next write index);
// everything about how a Transition is actually stored is delegated to a
// TransitionStorage backend, so swapping that backend never requires
// touching this class or any code that constructs/samples from it.
class ReplayBuffer {
public:
    // Capacity is derived from storage->capacity().
    explicit ReplayBuffer(std::unique_ptr<TransitionStorage> storage);

    // Inserts one transition, overwriting the oldest entry once capacity()
    // is reached.
    void add(Transition transition);

    // Draws `batch_size` transitions uniformly at random, with replacement,
    // using the caller-supplied RNG. The RNG is passed in rather than owned
    // (same reasoning as Space::sample in rl/core/space.hpp) so an entire
    // experiment's randomness -- environment stepping, network
    // initialization, buffer sampling -- can be driven from one seeded
    // stream the training loop controls, keeping the whole run
    // reproducible. Throws std::logic_error if the buffer is empty.
    TransitionBatch sample(size_t batch_size, std::mt19937& rng) const;

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

private:
    size_t capacity_;
    size_t size_ = 0;
    size_t next_index_ = 0;
    std::unique_ptr<TransitionStorage> storage_;
};

} // namespace rl::core
