#include "rl/core/trainer.hpp"

#include <algorithm>

#include "rl/core/transition.hpp"

namespace rl::core {

Trainer::Trainer(VectorEnvironment& train_env, Environment& eval_env, Agent& agent)
    : train_env_(train_env), eval_env_(eval_env), agent_(agent) {}

TrainingResult Trainer::train(size_t num_steps, std::optional<uint64_t> seed) {
    if (!train_env_reset_) {
        auto reset_result = train_env_.reset(seed);
        current_observations_ = std::move(reset_result.observations);
        episode_return_accumulators_.assign(train_env_.num_envs(), 0.0f);
        episode_length_accumulators_.assign(train_env_.num_envs(), 0);
        train_env_reset_ = true;
    }

    TrainingResult result;
    for (size_t step = 0; step < num_steps; ++step) {
        std::vector<Action> actions = agent_.act(current_observations_, /*explore=*/true);
        VectorStepResult step_result = train_env_.step(actions);

        std::vector<Transition> transitions =
            make_transitions(current_observations_, actions, step_result);
        agent_.observe_transitions(transitions);
        if (agent_.should_update()) {
            result.update_metrics.push_back(agent_.update());
        }

        for (size_t i = 0; i < train_env_.num_envs(); ++i) {
            episode_return_accumulators_[i] += step_result.rewards[i];
            ++episode_length_accumulators_[i];
            if (step_result.terminated[i] || step_result.truncated[i]) {
                result.episode_returns.push_back(episode_return_accumulators_[i]);
                result.episode_lengths.push_back(episode_length_accumulators_[i]);
                episode_return_accumulators_[i] = 0.0f;
                episode_length_accumulators_[i] = 0;
            }
        }

        current_observations_ = std::move(step_result.observations);
    }

    return result;
}

EvaluationResult Trainer::evaluate(size_t num_episodes, std::optional<uint64_t> seed,
                                    size_t max_steps_per_episode) {
    EvaluationResult result;
    result.episode_returns.reserve(num_episodes);
    result.episode_lengths.reserve(num_episodes);

    for (size_t episode = 0; episode < num_episodes; ++episode) {
        std::optional<uint64_t> episode_seed =
            seed.has_value() ? std::optional<uint64_t>(*seed + episode) : std::nullopt;
        auto reset_result = eval_env_.reset(episode_seed);
        Observation observation = std::move(reset_result.observation);

        float episode_return = 0.0f;
        size_t episode_length = 0;
        bool done = false;

        while (!done && episode_length < max_steps_per_episode) {
            std::vector<Action> actions =
                agent_.act(std::vector<Observation>{observation}, /*explore=*/false);
            StepResult step_result = eval_env_.step(actions.front());

            episode_return += step_result.reward;
            ++episode_length;
            done = step_result.terminated || step_result.truncated;
            observation = std::move(step_result.observation);
        }

        result.episode_returns.push_back(episode_return);
        result.episode_lengths.push_back(episode_length);
    }

    return result;
}

} // namespace rl::core
