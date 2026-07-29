#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "rl/core/types.hpp"

namespace rl::core {

// Space describes the shape/type/bounds of a set of valid Values -- either
// an observation space or an action space. It is deliberately minimal: it
// only answers "what does a valid value look like" (sample, contains,
// describe). It does NOT own an RNG, log anything, or know whether it is
// being used for observations or actions -- that context lives in the
// Environment that holds it.
//
// Why an RNG is passed in rather than owned by the Space:
// Reproducibility in RL requires that a whole rollout (env transitions,
// action sampling, network initialization, ...) be driven by a single
// seeded stream the caller controls. If Space owned its own hidden RNG,
// seeding an Environment would not deterministically seed the spaces it
// exposes, and "same seed -> same trajectory" would silently break the
// moment a policy calls observation_space().sample() for e.g. an epsilon
// random action.
class Space {
public:
    virtual ~Space() = default;

    // Draws a uniformly random valid value from this space using the
    // caller-supplied RNG.
    virtual Value sample(std::mt19937& rng) const = 0;

    // Returns true if `value` is a valid member of this space (correct
    // variant alternative, correct size, within bounds).
    virtual bool contains(const Value& value) const = 0;

    // Human-readable description, e.g. "Discrete(4)" or "Box(dim=2)".
    // Used for logging and error messages, not for equality checks.
    virtual std::string describe() const = 0;
};

// A discrete space of `n` values: valid indices are 0, 1, ..., n-1.
class Discrete final : public Space {
public:
    explicit Discrete(int64_t n);

    int64_t n() const noexcept { return n_; }

    Value sample(std::mt19937& rng) const override;
    bool contains(const Value& value) const override;
    std::string describe() const override;

private:
    int64_t n_;
};

// A continuous, axis-aligned box in R^dim, bounded elementwise by low/high.
// low.size() must equal high.size(); that size is the space's dimension.
class Box final : public Space {
public:
    Box(std::vector<float> low, std::vector<float> high);

    const std::vector<float>& low() const noexcept { return low_; }
    const std::vector<float>& high() const noexcept { return high_; }
    size_t dim() const noexcept { return low_.size(); }

    Value sample(std::mt19937& rng) const override;
    bool contains(const Value& value) const override;
    std::string describe() const override;

private:
    std::vector<float> low_;
    std::vector<float> high_;
};

} // namespace rl::core
