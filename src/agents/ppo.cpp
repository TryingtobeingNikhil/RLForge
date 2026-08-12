#include "rl/agents/ppo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "rl/tensor/autograd.hpp"

namespace rl::agents {
namespace {

PPOConfig validate_config(PPOConfig config) {
    if (config.state_dim <= 0 || config.num_actions <= 0 ||
        config.num_actions > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("PPO state_dim and num_actions must be positive");
    }
    if (!std::isfinite(config.learning_rate) || !(config.learning_rate > 0.0) ||
        !std::isfinite(config.gamma) || config.gamma < 0.0 ||
        config.gamma > 1.0 || config.gae_lambda < 0.0 ||
        !std::isfinite(config.gae_lambda) || config.gae_lambda > 1.0 ||
        !std::isfinite(config.clip_range) || !(config.clip_range > 0.0) ||
        !std::isfinite(config.value_loss_coefficient) ||
        config.value_loss_coefficient < 0.0 || config.entropy_coefficient < 0.0) {
        throw std::invalid_argument("PPO floating-point hyperparameters are invalid");
    }
    if (!std::isfinite(config.entropy_coefficient)) {
        throw std::invalid_argument("PPO floating-point hyperparameters are invalid");
    }
    for (int64_t width : config.hidden_dims) {
        if (width <= 0) {
            throw std::invalid_argument("PPO hidden dimensions must be positive");
        }
    }
    if (config.rollout_steps == 0 || config.update_epochs == 0 ||
        config.minibatch_size == 0) {
        throw std::invalid_argument("PPO rollout/update sizes must be positive");
    }
    return config;
}

std::vector<double> selected_rows(
    const std::vector<rl::core::Observation>& observations,
    const std::vector<size_t>& indices, int64_t state_dim) {
    if (indices.size() > std::numeric_limits<size_t>::max() /
                             static_cast<size_t>(state_dim)) {
        throw std::overflow_error("PPO observation batch is too large");
    }
    std::vector<double> data(indices.size() * static_cast<size_t>(state_dim));
    for (size_t row = 0; row < indices.size(); ++row) {
        if (indices[row] >= observations.size()) {
            throw std::out_of_range("PPO observation index is out of range");
        }
        const auto& values = rl::core::as_vector(observations[indices[row]]);
        if (static_cast<int64_t>(values.size()) != state_dim) {
            throw std::invalid_argument("PPO observation dimension mismatch");
        }
        for (int64_t col = 0; col < state_dim; ++col) {
            data[row * static_cast<size_t>(state_dim) + static_cast<size_t>(col)] =
                static_cast<double>(values[static_cast<size_t>(col)]);
        }
    }
    return data;
}

} // namespace

GAEOutput compute_gae(const std::vector<double>& rewards,
                      const std::vector<double>& values,
                      const std::vector<double>& next_values,
                      const rl::core::BoolArray& terminated,
                      const rl::core::BoolArray& truncated,
                      size_t time_steps, size_t num_envs,
                      double gamma, double gae_lambda) {
    if (num_envs != 0 &&
        time_steps > std::numeric_limits<size_t>::max() / num_envs) {
        throw std::overflow_error("compute_gae [T,N] size overflows");
    }
    const size_t total = time_steps * num_envs;
    if (time_steps == 0 || num_envs == 0 || rewards.size() != total ||
        values.size() != total || next_values.size() != total ||
        terminated.size() != total || truncated.size() != total) {
        throw std::invalid_argument("compute_gae expects equally sized non-empty [T,N] arrays");
    }
    if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0 ||
        !std::isfinite(gae_lambda) || gae_lambda < 0.0 || gae_lambda > 1.0) {
        throw std::invalid_argument("compute_gae gamma and lambda must be in [0,1]");
    }

    GAEOutput result;
    result.advantages.resize(total);
    result.returns.resize(total);
    std::vector<double> lane_advantage(num_envs, 0.0);
    for (size_t t = time_steps; t-- > 0;) {
        for (size_t n = 0; n < num_envs; ++n) {
            const size_t index = t * num_envs + n;
            const double bootstrap_mask = terminated[index] ? 0.0 : 1.0;
            const double trace_mask = (terminated[index] || truncated[index]) ? 0.0 : 1.0;
            const double delta = rewards[index] + gamma * bootstrap_mask *
                                                      next_values[index] - values[index];
            lane_advantage[n] = delta + gamma * gae_lambda * trace_mask *
                                            lane_advantage[n];
            result.advantages[index] = lane_advantage[n];
            result.returns[index] = lane_advantage[n] + values[index];
        }
    }
    return result;
}

PPOAgent::PPOAgent(PPOConfig config)
    : config_(validate_config(std::move(config))),
      network_(config_.state_dim, config_.hidden_dims, config_.num_actions,
               config_.seed),
      optimizer_(network_.parameters(), config_.learning_rate),
      rollout_(config_.rollout_steps),
      rng_(static_cast<std::mt19937::result_type>(config_.seed)) {}

rl::tensor::Tensor PPOAgent::observations_to_tensor(
    const std::vector<rl::core::Observation>& observations, int64_t state_dim) {
    if (observations.empty()) {
        throw std::invalid_argument("PPOAgent::act requires at least one observation");
    }
    if (observations.size() >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("PPO observation batch is too large");
    }
    std::vector<size_t> indices(observations.size());
    std::iota(indices.begin(), indices.end(), size_t{0});
    return rl::tensor::Tensor::from_data(
        selected_rows(observations, indices, state_dim),
        {static_cast<int64_t>(observations.size()), state_dim});
}

std::vector<rl::core::Action> PPOAgent::act(
    const std::vector<rl::core::Observation>& observations, bool explore) {
    if (explore && pending_actions_) {
        throw std::logic_error(
            "PPOAgent::act called twice before observe_transitions consumed the actions");
    }

    std::vector<rl::core::Action> actions;
    PendingActionBatch pending;
    actions.reserve(observations.size());
    pending.actions.reserve(observations.size());
    pending.log_probabilities.reserve(observations.size());
    pending.values.reserve(observations.size());

    auto guard = rl::tensor::no_grad();
    auto output = network_.forward(observations_to_tensor(observations, config_.state_dim));
    auto log_probabilities = output.policy_logits.log_softmax();
    auto value_indices = rl::tensor::Tensor::zeros(
        {static_cast<int64_t>(observations.size())});
    auto values = output.values.gather(value_indices);

    for (size_t row = 0; row < observations.size(); ++row) {
        const int64_t base = static_cast<int64_t>(row) * config_.num_actions;
        int64_t action = 0;
        if (explore) {
            std::vector<double> weights(static_cast<size_t>(config_.num_actions));
            for (int64_t col = 0; col < config_.num_actions; ++col) {
                weights[static_cast<size_t>(col)] =
                    std::exp(log_probabilities[base + col]);
            }
            std::discrete_distribution<int64_t> distribution(weights.begin(),
                                                              weights.end());
            action = distribution(rng_);
        } else {
            double best = output.policy_logits[base];
            for (int64_t col = 1; col < config_.num_actions; ++col) {
                if (output.policy_logits[base + col] > best) {
                    best = output.policy_logits[base + col];
                    action = col;
                }
            }
        }
        actions.emplace_back(action);
        if (explore) {
            pending.actions.emplace_back(action);
            pending.log_probabilities.push_back(log_probabilities[base + action]);
            pending.values.push_back(values[static_cast<int64_t>(row)]);
        }
    }

    if (explore) pending_actions_ = std::move(pending);
    return actions;
}

void PPOAgent::observe_transitions(
    const std::vector<rl::core::Transition>& transitions) {
    if (!pending_actions_) {
        throw std::logic_error(
            "PPOAgent::observe_transitions requires a preceding exploratory act call");
    }
    if (transitions.size() != pending_actions_->actions.size()) {
        throw std::invalid_argument("PPO transition batch width does not match act batch");
    }
    for (size_t i = 0; i < transitions.size(); ++i) {
        if (transitions[i].action != pending_actions_->actions[i]) {
            throw std::invalid_argument(
                "PPO transition action differs from sampled action");
        }
    }

    rl::replay_buffers::RolloutStep step;
    step.log_probabilities = std::move(pending_actions_->log_probabilities);
    step.values = std::move(pending_actions_->values);
    step.observations.reserve(transitions.size());
    step.actions.reserve(transitions.size());
    step.rewards.reserve(transitions.size());
    step.next_observations.reserve(transitions.size());
    step.terminated.reserve(transitions.size());
    step.truncated.reserve(transitions.size());

    for (size_t i = 0; i < transitions.size(); ++i) {
        step.observations.push_back(transitions[i].observation);
        step.actions.push_back(transitions[i].action);
        step.rewards.push_back(transitions[i].reward);
        step.next_observations.push_back(transitions[i].next_observation);
        step.terminated.push_back(transitions[i].terminated ? 1 : 0);
        step.truncated.push_back(transitions[i].truncated ? 1 : 0);
    }
    pending_actions_.reset();
    rollout_.append(std::move(step));
}

bool PPOAgent::should_update() const { return rollout_.ready(); }

rl::core::Metrics PPOAgent::update() {
    if (!rollout_.ready()) {
        throw std::logic_error("PPOAgent::update requires a full rollout");
    }

    const size_t T = rollout_.size();
    const size_t N = rollout_.num_envs();
    if (N != 0 && T > std::numeric_limits<size_t>::max() / N) {
        throw std::overflow_error("PPO rollout size overflows");
    }
    const size_t total = T * N;
    if (total > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("PPO rollout is too large for tensor shapes");
    }
    std::vector<rl::core::Observation> observations;
    std::vector<rl::core::Observation> next_observations;
    std::vector<double> actions(total), rewards(total), old_log_probs(total),
        old_values(total), advantages(total), returns(total), next_values(total);
    rl::core::BoolArray terminated(total), truncated(total);
    observations.reserve(total);
    next_observations.reserve(total);

    for (size_t t = 0; t < T; ++t) {
        const auto& step = rollout_.steps()[t];
        for (size_t n = 0; n < N; ++n) {
            const size_t index = t * N + n;
            observations.push_back(step.observations[n]);
            next_observations.push_back(step.next_observations[n]);
            actions[index] = static_cast<double>(rl::core::as_index(step.actions[n]));
            rewards[index] = static_cast<double>(step.rewards[n]);
            old_log_probs[index] = step.log_probabilities[n];
            old_values[index] = step.values[n];
            terminated[index] = step.terminated[n];
            truncated[index] = step.truncated[n];
        }
    }

    {
        auto guard = rl::tensor::no_grad();
        auto output = network_.forward(
            observations_to_tensor(next_observations, config_.state_dim));
        auto indices = rl::tensor::Tensor::zeros({static_cast<int64_t>(total)});
        next_values = output.values.gather(indices).data();
    }

    auto gae = compute_gae(rewards, old_values, next_values, terminated, truncated,
                           T, N, config_.gamma, config_.gae_lambda);
    advantages = std::move(gae.advantages);
    returns = std::move(gae.returns);

    if (config_.normalize_advantages && total > 1) {
        const double mean = std::accumulate(advantages.begin(), advantages.end(), 0.0) /
                            static_cast<double>(total);
        double variance = 0.0;
        for (double value : advantages) variance += (value - mean) * (value - mean);
        variance /= static_cast<double>(total);
        const double scale = 1.0 / std::sqrt(variance + 1e-8);
        for (double& value : advantages) value = (value - mean) * scale;
    }

    std::vector<size_t> order(total);
    std::iota(order.begin(), order.end(), size_t{0});
    double policy_loss_sum = 0.0;
    double value_loss_sum = 0.0;
    double entropy_sum = 0.0;
    double approximate_kl_sum = 0.0;
    double clip_fraction_sum = 0.0;
    size_t update_count = 0;

    for (size_t epoch = 0; epoch < config_.update_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng_);
        for (size_t begin = 0; begin < total; begin += config_.minibatch_size) {
            const size_t end = std::min(total, begin + config_.minibatch_size);
            std::vector<size_t> indices(order.begin() + static_cast<std::ptrdiff_t>(begin),
                                        order.begin() + static_cast<std::ptrdiff_t>(end));
            const int64_t B = static_cast<int64_t>(indices.size());
            std::vector<double> action_batch(indices.size());
            std::vector<double> old_log_prob_batch(indices.size());
            std::vector<double> advantage_batch(indices.size());
            std::vector<double> return_batch(indices.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                action_batch[i] = actions[indices[i]];
                old_log_prob_batch[i] = old_log_probs[indices[i]];
                advantage_batch[i] = advantages[indices[i]];
                return_batch[i] = returns[indices[i]];
            }

            auto states = rl::tensor::Tensor::from_data(
                selected_rows(observations, indices, config_.state_dim),
                {B, config_.state_dim});
            auto action_tensor = rl::tensor::Tensor::from_data(std::move(action_batch), {B});
            auto old_log_prob_tensor =
                rl::tensor::Tensor::from_data(std::move(old_log_prob_batch), {B});
            auto advantage_tensor =
                rl::tensor::Tensor::from_data(std::move(advantage_batch), {B});
            auto return_tensor =
                rl::tensor::Tensor::from_data(std::move(return_batch), {B});

            auto output = network_.forward(states);
            auto all_log_probs = output.policy_logits.log_softmax();
            auto new_log_probs = all_log_probs.gather(action_tensor);
            auto ratio = new_log_probs.sub(old_log_prob_tensor).exp();
            auto unclipped = ratio.mul(advantage_tensor);
            auto clipped = ratio.clamp(1.0 - config_.clip_range,
                                       1.0 + config_.clip_range)
                               .mul(advantage_tensor);
            auto policy_loss = unclipped.minimum(clipped).mean().mul(-1.0);
            auto entropy = all_log_probs.exp().mul(all_log_probs).mean().mul(
                -static_cast<double>(config_.num_actions));
            auto value_indices = rl::tensor::Tensor::zeros({B});
            auto predicted_values = output.values.gather(value_indices);
            auto value_loss = predicted_values.sub(return_tensor).square().mean();
            auto total_loss = policy_loss
                                  .add(value_loss.mul(config_.value_loss_coefficient))
                                  .sub(entropy.mul(config_.entropy_coefficient));

            double approximate_kl = 0.0;
            double clip_fraction = 0.0;
            for (int64_t i = 0; i < B; ++i) {
                approximate_kl += old_log_probs[indices[static_cast<size_t>(i)]] -
                                  new_log_probs[i];
                if (std::abs(ratio[i] - 1.0) > config_.clip_range) {
                    clip_fraction += 1.0;
                }
            }
            approximate_kl /= static_cast<double>(B);
            clip_fraction /= static_cast<double>(B);

            optimizer_.zero_grad();
            total_loss.backward();
            optimizer_.step();

            policy_loss_sum += policy_loss.item();
            value_loss_sum += value_loss.item();
            entropy_sum += entropy.item();
            approximate_kl_sum += approximate_kl;
            clip_fraction_sum += clip_fraction;
            ++update_count;
        }
    }

    rollout_.clear();
    const double divisor = static_cast<double>(update_count);
    return rl::core::Metrics{
        {"policy_loss", policy_loss_sum / divisor},
        {"value_loss", value_loss_sum / divisor},
        {"entropy", entropy_sum / divisor},
        {"approximate_kl", approximate_kl_sum / divisor},
        {"clip_fraction", clip_fraction_sum / divisor},
        {"minibatch_updates", static_cast<int64_t>(update_count)}};
}

} // namespace rl::agents
