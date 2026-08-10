#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "rl/core/transition.hpp"
#include "rl/core/types.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::data {

// ---------------------------------------------------------------------------
// TensorBatch — the Tensor representation of a sampled TransitionBatch.
//
// This struct is algorithm-agnostic. DQN uses gather() on actions; PPO/SAC
// will process fields differently. Nothing in this struct is DQN-specific.
// ---------------------------------------------------------------------------
struct TensorBatch {
    rl::tensor::Tensor states;       // [B, state_dim]  — float, no requires_grad
    rl::tensor::Tensor actions;      // [B]             — int action index as double
    rl::tensor::Tensor rewards;      // [B]             — float rewards
    rl::tensor::Tensor next_states;  // [B, state_dim]  — float, no requires_grad
    rl::tensor::Tensor terminated;   // [B]             — 1.0=terminated, 0.0=not
};

// ---------------------------------------------------------------------------
// batch_to_tensors — convert a ReplayBuffer sample into contiguous Tensors.
//
// Parameters:
//   batch     — TransitionBatch returned by ReplayBuffer::sample().
//               This struct owns no Tensors; it is purely C++ standard types.
//   state_dim — expected size of each Observation vector<float>.
//
// Returns a TensorBatch with freshly allocated Tensors (no requires_grad).
// The caller (DQN's train_step) sets up grad tracking on whatever sub-result
// needs it (e.g. q_values = online_net.forward(states).gather(actions)).
//
// TERMINATED vs TRUNCATED (CRITICAL CORRECTNESS):
//   terminated[i] = 1.0  iff  batch.terminated[i] is true.
//   truncated transitions DO NOT affect the terminated mask.
//
//   Bellman bootstrap mask = (1 - terminated).
//   A truncated episode was cut off by a time limit, NOT by a true terminal
//   MDP state. The underlying MDP continues past the truncation boundary, so
//   the value estimate at next_state is still valid and must NOT be zeroed.
//   Conflating terminated and truncated (the classic "done" flag bug) silently
//   corrupts every value target at a truncation boundary.
//
// ACTION FORMAT:
//   Actions are expected to be Discrete (int64_t). They are cast to double
//   because Tensor's scalar type is double. DQN's train_step passes the
//   actions Tensor to gather(), which re-casts each element to int64_t.
//
// STATE FORMAT:
//   Each Observation must be std::vector<float> (Box space) of size state_dim.
//   Throws std::invalid_argument for wrong type or wrong size.
// ---------------------------------------------------------------------------
inline TensorBatch batch_to_tensors(const rl::core::TransitionBatch& batch,
                                    int64_t state_dim) {
    const int64_t B = static_cast<int64_t>(batch.rewards.size());
    if (B == 0) {
        throw std::invalid_argument("batch_to_tensors: batch is empty.");
    }

    std::vector<double> states_data(static_cast<size_t>(B * state_dim));
    std::vector<double> actions_data(static_cast<size_t>(B));
    std::vector<double> rewards_data(static_cast<size_t>(B));
    std::vector<double> next_states_data(static_cast<size_t>(B * state_dim));
    std::vector<double> terminated_data(static_cast<size_t>(B));

    for (int64_t i = 0; i < B; ++i) {
        const size_t si = static_cast<size_t>(i);

        // ---- states --------------------------------------------------------
        if (!std::holds_alternative<std::vector<float>>(batch.observations[si])) {
            throw std::invalid_argument(
                "batch_to_tensors: observation[" + std::to_string(i) +
                "] is not vector<float>. batch_to_tensors expects Box observations.");
        }
        const auto& obs_vec = std::get<std::vector<float>>(batch.observations[si]);
        if (static_cast<int64_t>(obs_vec.size()) != state_dim) {
            throw std::invalid_argument(
                "batch_to_tensors: observation[" + std::to_string(i) +
                "] has size " + std::to_string(obs_vec.size()) +
                ", expected state_dim=" + std::to_string(state_dim) + ".");
        }
        for (int64_t j = 0; j < state_dim; ++j) {
            states_data[static_cast<size_t>(i * state_dim + j)] =
                static_cast<double>(obs_vec[static_cast<size_t>(j)]);
        }

        // ---- actions -------------------------------------------------------
        if (!std::holds_alternative<int64_t>(batch.actions[si])) {
            throw std::invalid_argument(
                "batch_to_tensors: action[" + std::to_string(i) +
                "] is not int64_t. batch_to_tensors expects Discrete actions.");
        }
        actions_data[si] = static_cast<double>(std::get<int64_t>(batch.actions[si]));

        // ---- rewards -------------------------------------------------------
        rewards_data[si] = static_cast<double>(batch.rewards[si]);

        // ---- next_states ---------------------------------------------------
        if (!std::holds_alternative<std::vector<float>>(batch.next_observations[si])) {
            throw std::invalid_argument(
                "batch_to_tensors: next_observation[" + std::to_string(i) +
                "] is not vector<float>.");
        }
        const auto& nxt_vec =
            std::get<std::vector<float>>(batch.next_observations[si]);
        if (static_cast<int64_t>(nxt_vec.size()) != state_dim) {
            throw std::invalid_argument(
                "batch_to_tensors: next_observation[" + std::to_string(i) +
                "] has size " + std::to_string(nxt_vec.size()) +
                ", expected state_dim=" + std::to_string(state_dim) + ".");
        }
        for (int64_t j = 0; j < state_dim; ++j) {
            next_states_data[static_cast<size_t>(i * state_dim + j)] =
                static_cast<double>(nxt_vec[static_cast<size_t>(j)]);
        }

        // ---- terminated mask (NOT truncated) --------------------------------
        // ONLY batch.terminated is used. batch.truncated is intentionally ignored.
        terminated_data[si] = batch.terminated[si] ? 1.0 : 0.0;
    }

    return TensorBatch{
        rl::tensor::Tensor::from_data(std::move(states_data),      {B, state_dim}),
        rl::tensor::Tensor::from_data(std::move(actions_data),     {B}),
        rl::tensor::Tensor::from_data(std::move(rewards_data),     {B}),
        rl::tensor::Tensor::from_data(std::move(next_states_data), {B, state_dim}),
        rl::tensor::Tensor::from_data(std::move(terminated_data),  {B})
    };
}

}  // namespace rl::data
