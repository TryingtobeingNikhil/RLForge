#pragma once

#include <cstdint>
#include <memory>
#include <random>

#include "rl/core/environment.hpp"
#include "rl/core/space.hpp"
#include "rl/core/types.hpp"

namespace rl::envs {

// Deliberately a free (non-nested) struct rather than GridWorld::Config
// defined in-class: a constructor default argument of `= {}` for a type
// nested inside the very class being defined can fail to resolve during
// class-body parsing on standards-conforming compilers (reproducible on
// GCC 13/14). Defining it at namespace scope and aliasing it as
// GridWorld::Config below keeps the public API identical while avoiding
// the issue entirely.
struct GridWorldConfig {
    int64_t size = 5;
    int64_t max_episode_steps = 100;
    float slip_probability = 0.1f;
};

// GridWorld is the reference environment for this milestone: it exists to
// prove the Environment/Space contract is actually usable end to end, not
// to be an interesting RL benchmark.
//
// Layout: an `size` x `size` grid. The agent starts at (0, 0) and the goal
// is at (size-1, size-1). Actions are Discrete(4): Up, Down, Left, Right.
// The observation is a Box(dim=2) of the agent's (x, y) coordinates.
//
// Reward: -1 per step (encourages shortest paths), +10 on reaching the goal.
// terminated=true when the agent reaches the goal (a true terminal state of
// this MDP). truncated=true when `max_episode_steps` is exceeded without
// reaching the goal (a time limit external to the MDP itself) -- this is the
// concrete case the terminated/truncated split exists for.
//
// `slip_probability` gives a chance that the requested action is replaced by
// a uniformly random one. This is not for realism -- it exists so that
// "same seed -> same trajectory" is a meaningful test rather than trivially
// true of a fully deterministic environment.
class GridWorld final : public rl::core::EnvironmentBase {
public:
    enum class GridAction : int64_t { Up = 0, Down = 1, Left = 2, Right = 3 };

    using Config = GridWorldConfig;

    explicit GridWorld(Config config = {});

    const rl::core::Space& observation_space() const override {
        return *observation_space_;
    }
    const rl::core::Space& action_space() const override {
        return *action_space_;
    }
    std::string name() const override { return "GridWorld"; }

    // Convenience accessors, mainly useful for tests.
    int64_t agent_x() const noexcept { return agent_x_; }
    int64_t agent_y() const noexcept { return agent_y_; }

protected:
    rl::core::ResetResult reset_impl(std::optional<uint64_t> seed) override;
    rl::core::StepResult step_impl(const rl::core::Action& action) override;

private:
    rl::core::Observation current_observation() const;
    int64_t clamp(int64_t value) const;

    Config config_;
    std::unique_ptr<rl::core::Space> observation_space_;
    std::unique_ptr<rl::core::Space> action_space_;
    std::mt19937 rng_;
    int64_t agent_x_ = 0;
    int64_t agent_y_ = 0;
    int64_t step_count_ = 0;
};

} // namespace rl::envs
