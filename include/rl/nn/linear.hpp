#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rl/nn/module.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::nn {

// ---------------------------------------------------------------------------
// Linear — fully-connected (affine) layer.
//
//   y = x @ W^T + b
//
// where:
//   x has shape [B, in_features]  (batch of B inputs, each of size in_features)
//   W has shape [out_features, in_features]
//   b has shape [out_features]
//   y has shape [B, out_features]
//
// Weight initialisation: He (Kaiming) normal — values drawn from N(0, σ²)
//   with σ = sqrt(2 / in_features). This is appropriate for layers followed
//   by ReLU activations and provides good gradient flow at network initialisation.
//
// Bias initialisation: zero.
//
// The forward pass computes:
//   1. x.matmul(weight_->transpose())  — [B,in] @ [in,out] = [B,out]
//   2. result + *bias_                 — [B,out] + [out] (broadcasting add)
//
// Transpose decision (Milestone 6):
//   We store W as [out,in] and call W.transpose() at forward time (producing
//   [in,out]) for the matmul. This is cleaner than storing W pre-transposed
//   because W's gradient computed by matmul backward is in [out,in] shape —
//   exactly matching how W is stored, so no extra transposition is needed in
//   the parameter update.
// ---------------------------------------------------------------------------
class Linear : public Module {
public:
    // Constructs a Linear layer with He-initialised weights and zero biases.
    // Both weight_ and bias_ have requires_grad=true.
    Linear(int64_t in_features, int64_t out_features);

    // forward: x must have shape [B, in_features].
    // Returns a tensor of shape [B, out_features].
    rl::tensor::Tensor forward(const rl::tensor::Tensor& input) override;

    // Returns {weight_, bias_} in that order.
    // Both are shared_ptr<Tensor> with requires_grad()==true.
    std::vector<std::shared_ptr<rl::tensor::Tensor>> parameters() const override;

    // Direct accessors for testing.
    const rl::tensor::Tensor& weight() const { return *weight_; }
    const rl::tensor::Tensor& bias()   const { return *bias_;   }

private:
    std::shared_ptr<rl::tensor::Tensor> weight_;  // shape [out_features, in_features]
    std::shared_ptr<rl::tensor::Tensor> bias_;    // shape [out_features]
};

} // namespace rl::nn
