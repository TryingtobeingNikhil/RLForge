#include "rl/vector_envs/sync_vector_environment.hpp"

#include <stdexcept>
#include <utility>

#include "rl/core/space.hpp"

namespace rl::vector_envs {

using rl::core::spaces_compatible;

SyncVectorEnvironment::SyncVectorEnvironment(std::vector<EnvFactory> factories) {
    if (factories.empty()) {
        throw std::invalid_argument(
            "SyncVectorEnvironment requires at least one env factory");
    }

    envs_.reserve(factories.size());
    for (size_t i = 0; i < factories.size(); ++i) {
        auto& factory = factories[i];
        if (!factory) {
            throw std::invalid_argument(
                "SyncVectorEnvironment factory " + std::to_string(i) +
                " is empty");
        }
        auto env = factory();
        if (!env) {
            throw std::invalid_argument(
                "SyncVectorEnvironment factory " + std::to_string(i) +
                " returned a null environment");
        }
        envs_.push_back(std::move(env));
    }

    const rl::core::Space& reference_obs_space = envs_.front()->observation_space();
    const rl::core::Space& reference_action_space = envs_.front()->action_space();
    for (size_t i = 1; i < envs_.size(); ++i) {
        if (!spaces_compatible(reference_obs_space, envs_[i]->observation_space())) {
            throw std::invalid_argument(
                "SyncVectorEnvironment requires all sub-envs to share the "
                "same observation_space; sub-env 0 has " +
                reference_obs_space.describe() + " but sub-env " +
                std::to_string(i) + " has " +
                envs_[i]->observation_space().describe());
        }
        if (!spaces_compatible(reference_action_space, envs_[i]->action_space())) {
            throw std::invalid_argument(
                "SyncVectorEnvironment requires all sub-envs to share the "
                "same action_space; sub-env 0 has " +
                reference_action_space.describe() + " but sub-env " +
                std::to_string(i) + " has " +
                envs_[i]->action_space().describe());
        }
    }
}

rl::core::VectorResetResult SyncVectorEnvironment::reset_impl(
    std::optional<uint64_t> seed) {
    rl::core::VectorResetResult result;
    result.observations.reserve(envs_.size());
    result.infos.reserve(envs_.size());

    for (size_t i = 0; i < envs_.size(); ++i) {
        // Derive a distinct seed per sub-env from the shared base seed, so
        // sub-envs don't produce correlated trajectories -- see
        // VectorEnvironment::reset() documentation.
        std::optional<uint64_t> sub_seed =
            seed.has_value() ? std::optional<uint64_t>(*seed + i) : std::nullopt;
        auto reset_result = envs_[i]->reset(sub_seed);
        result.observations.push_back(std::move(reset_result.observation));
        result.infos.push_back(std::move(reset_result.info));
    }

    return result;
}

rl::core::VectorStepResult SyncVectorEnvironment::step_impl(
    const std::vector<rl::core::Action>& actions) {
    if (actions.size() != envs_.size()) {
        throw std::invalid_argument(
            "SyncVectorEnvironment::step() received " +
            std::to_string(actions.size()) + " actions for " +
            std::to_string(envs_.size()) + " sub-envs");
    }

    const size_t n = envs_.size();
    rl::core::VectorStepResult result;
    result.observations.resize(n);
    result.rewards.resize(n);
    result.terminated.resize(n);
    result.truncated.resize(n);
    result.infos.resize(n);
    result.final_observations.resize(n);

    for (size_t i = 0; i < n; ++i) {
        auto step_result = envs_[i]->step(actions[i]);
        const bool done = step_result.terminated || step_result.truncated;

        result.rewards[i] = step_result.reward;
        result.terminated[i] = step_result.terminated ? 1 : 0;
        result.truncated[i] = step_result.truncated ? 1 : 0;
        result.infos[i] = std::move(step_result.info);

        if (done) {
            // Preserve the true terminal observation before overwriting
            // observations[i] with the post-auto-reset initial observation.
            // No seed is passed here -- auto-reset continues this sub-env's
            // own RNG stream rather than reseeding it, matching what
            // Environment::reset(std::nullopt) means for a single env.
            result.final_observations[i] = std::move(step_result.observation);
            auto reset_result = envs_[i]->reset();
            result.observations[i] = std::move(reset_result.observation);
        } else {
            result.final_observations[i] = std::nullopt;
            result.observations[i] = std::move(step_result.observation);
        }
    }

    return result;
}

} // namespace rl::vector_envs
