#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rl/nn/linear.hpp"
#include "rl/nn/module.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::nn {

// ---------------------------------------------------------------------------
// QNetwork — multi-layer fully-connected network producing raw Q-values.
//
// Architecture: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
// The FINAL layer has NO activation: Q-values are unbounded scalars and must
// not be clamped by ReLU. Adding a final activation is a common mistake that
// prevents the network from representing negative Q-values.
//
// BATCHED INPUT ONLY:
//   forward() requires shape [B, input_dim] where B >= 1.
//   Unbatched single-state input (shape [input_dim]) is NOT supported. The
//   caller must reshape: Tensor::from_data(obs, {1, input_dim}).
//   A runtime assertion throws std::invalid_argument if ndim() != 2.
//   Rationale: Linear::forward() already requires [B, in_features] (enforced
//   by matmul). Making QNetwork silently reshape would hide shape errors from
//   callers and create subtle shape bugs in batched training code.
//
// PARAMETERS:
//   parameters() returns the parameters of ALL Linear layers in forward order:
//     { layer0.weight, layer0.bias, layer1.weight, layer1.bias, ... }
//   Total count: 2 * num_layers = 2 * (hidden_dims.size() + 1).
//
// REQUIRES_GRAD:
//   By default (online network): all parameters have requires_grad=true.
//   For a target network: after construction, iterate parameters() and call
//   p->requires_grad_(false) on each. DQNAgent does this automatically.
// ---------------------------------------------------------------------------
class QNetwork : public Module {
public:
    // Constructs a QNetwork.
    //   input_dim   — observation feature dimension
    //   hidden_dims — widths of hidden layers (may be empty for a single-layer net)
    //   output_dim  — number of actions (Q-value output dimension)
    //
    // Example: QNetwork(4, {64, 64}, 2) builds:
    //   Linear(4->64) -> ReLU -> Linear(64->64) -> ReLU -> Linear(64->2)
    QNetwork(int64_t input_dim, std::vector<int64_t> hidden_dims,
             int64_t output_dim,
             std::optional<uint64_t> seed = std::nullopt);

    // forward: input must have shape [B, input_dim].
    // Returns shape [B, output_dim] (raw Q-values, no final activation).
    // Throws std::invalid_argument if input.ndim() != 2.
    rl::tensor::Tensor forward(const rl::tensor::Tensor& input) override;

    // Returns all parameter tensors from all Linear layers, in forward order:
    //   { layer0.weight, layer0.bias, layer1.weight, layer1.bias, ... }
    // The returned shared_ptr objects are stable (same pointers on every call).
    // Optimizers that hold references to parameters rely on this stability.
    std::vector<std::shared_ptr<rl::tensor::Tensor>> parameters() const override;

private:
    // All Linear layers in forward order. The i-th layer is followed by ReLU
    // for all i < layers_.size()-1; the last layer has no activation.
    std::vector<Linear> layers_;
};

} // namespace rl::nn
