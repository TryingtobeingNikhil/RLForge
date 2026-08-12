#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "rl/nn/qnetwork.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::nn {

struct ActorCriticOutput {
    rl::tensor::Tensor policy_logits; // [B, num_actions]
    rl::tensor::Tensor values;        // [B, 1]
};

// Separate actor and critic MLPs. Keeping their parameters independent makes
// the first PPO implementation explicit and avoids hidden gradient coupling.
class ActorCriticNetwork {
public:
    ActorCriticNetwork(int64_t input_dim, std::vector<int64_t> hidden_dims,
                       int64_t num_actions,
                       std::optional<uint64_t> seed = std::nullopt);

    ActorCriticOutput forward(const rl::tensor::Tensor& observations);
    std::vector<std::shared_ptr<rl::tensor::Tensor>> parameters() const;

private:
    QNetwork actor_;
    QNetwork critic_;
};

} // namespace rl::nn
