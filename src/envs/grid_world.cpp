#include "rl/envs/grid_world.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rl::envs {

using rl::core::Discrete;
using rl::core::Box;

GridWorld::GridWorld(Config config) : config_(config), rng_(std::random_device{}()) {
    if (config_.size < 2) {
        throw std::invalid_argument("GridWorld size must be >= 2, got " +
                                     std::to_string(config_.size));
    }
    if (config_.max_episode_steps <= 0) {
        throw std::invalid_argument("GridWorld max_episode_steps must be positive");
    }
    if (!std::isfinite(config_.slip_probability) ||
        config_.slip_probability < 0.0f || config_.slip_probability > 1.0f) {
        throw std::invalid_argument(
            "GridWorld slip_probability must be in [0, 1]");
    }
    const float max_coord = static_cast<float>(config_.size - 1);
    observation_space_ = std::make_unique<Box>(
        std::vector<float>{0.0f, 0.0f}, std::vector<float>{max_coord, max_coord});
    action_space_ = std::make_unique<Discrete>(4);
}

int64_t GridWorld::clamp(int64_t value) const {
    return std::clamp<int64_t>(value, 0, config_.size - 1);
}

rl::core::Observation GridWorld::current_observation() const {
    return rl::core::Observation{std::vector<float>{
        static_cast<float>(agent_x_), static_cast<float>(agent_y_)}};
}

rl::core::ResetResult GridWorld::reset_impl(std::optional<uint64_t> seed) {
    if (seed.has_value()) {
        rng_.seed(static_cast<std::mt19937::result_type>(*seed));
    }
    agent_x_ = 0;
    agent_y_ = 0;
    step_count_ = 0;
    return rl::core::ResetResult{current_observation(), {}};
}

rl::core::StepResult GridWorld::step_impl(const rl::core::Action& action) {
    if (!action_space_->contains(action)) {
        throw std::invalid_argument(
            "GridWorld received an action outside its action_space " +
            action_space_->describe());
    }

    int64_t requested = rl::core::as_index(action);

    // Slip: with probability slip_probability, override the requested
    // action with a uniformly random one drawn from this env's own RNG
    // stream (seeded via reset()), so trajectories stay reproducible.
    std::uniform_real_distribution<float> slip_roll(0.0f, 1.0f);
    if (slip_roll(rng_) < config_.slip_probability) {
        std::uniform_int_distribution<int64_t> random_action(0, 3);
        requested = random_action(rng_);
    }

    switch (static_cast<GridAction>(requested)) {
        case GridAction::Up:
            agent_y_ = clamp(agent_y_ + 1);
            break;
        case GridAction::Down:
            agent_y_ = clamp(agent_y_ - 1);
            break;
        case GridAction::Left:
            agent_x_ = clamp(agent_x_ - 1);
            break;
        case GridAction::Right:
            agent_x_ = clamp(agent_x_ + 1);
            break;
    }

    ++step_count_;

    const bool reached_goal =
        (agent_x_ == config_.size - 1) && (agent_y_ == config_.size - 1);
    const bool terminated = reached_goal;
    const bool truncated =
        !terminated && step_count_ >= config_.max_episode_steps;
    const float reward = terminated ? 10.0f : -1.0f;

    return rl::core::StepResult{current_observation(), reward, terminated,
                                 truncated, {}};
}

} // namespace rl::envs
