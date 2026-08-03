#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "rl/core/agent.hpp"
#include "rl/core/environment.hpp"
#include "rl/core/vector_environment.hpp"

namespace rl::core {

// Result of a Trainer::train() call: episodes that completed during this
// call (each may have ended via terminated or truncated -- both count as
// "completed" for accounting purposes), and the metrics returned by every
// update() call the agent performed. Not a full logging/metrics framework
// (that's a later milestone) -- just the raw series, for the caller to
// print, plot, or feed to whatever logging arrives later.
struct TrainingResult {
    std::vector<float> episode_returns;
    std::vector<size_t> episode_lengths;
    std::vector<Metrics> update_metrics;
};

// Result of a Trainer::evaluate() call: one entry per completed episode.
struct EvaluationResult {
    std::vector<float> episode_returns;
    std::vector<size_t> episode_lengths;
};

// Trainer is the generic training loop: it drives an Agent against a
// VectorEnvironment for training and a separate (non-vectorized)
// Environment for evaluation, using only the Agent and
// (Vector)Environment interfaces. It has no knowledge of which concrete
// algorithm or environment it's driving -- tabular Q-Learning is the first
// Agent implementation exercised through it, not a special case it's built
// around.
//
// Training is driven through VectorEnvironment -- even when num_envs() is
// 1, as in the simplest tabular Q-Learning setup -- rather than through a
// separate single-env code path, so this loop never needs to change when
// parallel rollout collection or a multi-env DQN/PPO setup arrives later.
//
// Evaluation uses a separate, non-vectorized Environment rather than
// reusing the training env: stepping the training env for evaluation
// purposes would disturb whatever in-flight episode training is in the
// middle of collecting.
class Trainer {
public:
    Trainer(VectorEnvironment& train_env, Environment& eval_env, Agent& agent);

    // Runs exactly `num_steps` steps of the training VectorEnvironment
    // (one "step" = one call to train_env.step(), advancing every sub-env
    // once). The training environment is reset on the very first call to
    // train(); subsequent calls continue from wherever training left off,
    // so train() can be called repeatedly (e.g. interleaved with
    // evaluate() for periodic evaluation) without losing in-flight
    // episodes.
    TrainingResult train(size_t num_steps, std::optional<uint64_t> seed = std::nullopt);

    // Runs `num_episodes` complete episodes in the evaluation Environment
    // with exploration disabled (agent.act(..., /*explore=*/false)). Does
    // NOT call agent.observe_transitions() or agent.update() -- evaluation
    // only reads the agent's current policy, never mutates its state.
    // `max_steps_per_episode` is a defensive cap in case an environment or
    // policy combination never naturally terminates/truncates.
    EvaluationResult evaluate(size_t num_episodes,
                               std::optional<uint64_t> seed = std::nullopt,
                               size_t max_steps_per_episode = 100000);

private:
    VectorEnvironment& train_env_;
    Environment& eval_env_;
    Agent& agent_;

    bool train_env_reset_ = false;
    std::vector<Observation> current_observations_;
    std::vector<float> episode_return_accumulators_;
    std::vector<size_t> episode_length_accumulators_;
};

} // namespace rl::core
