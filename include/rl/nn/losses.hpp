#pragma once

#include "rl/tensor/tensor.hpp"

namespace rl::nn {

// ---------------------------------------------------------------------------
// mse_loss — Mean Squared Error loss function.
//
//   mse_loss(pred, target) = mean((pred - target)^2)
//
// Implemented as pure composition of existing ops:
//   sub → square → mean
//
// No custom backward logic is needed — the backward pass is handled entirely
// by the autograd graph built by sub(), square(), and mean().
//
// pred and target must have exactly the same shape. target is not expected
// to require gradients (it is a ground-truth label), but the function works
// even if target.requires_grad() is true.
//
// Returns a scalar Tensor (shape {}), suitable for calling .backward() on.
//
// Location rationale: losses.hpp lives alongside module.hpp and linear.hpp
// in include/rl/nn/ because loss functions are part of the neural network
// training infrastructure, not the tensor primitive layer. Future losses
// (cross_entropy, huber) will live here too.
// ---------------------------------------------------------------------------
inline rl::tensor::Tensor mse_loss(const rl::tensor::Tensor& pred,
                                   const rl::tensor::Tensor& target) {
    return pred.sub(target).square().mean();
}

} // namespace rl::nn
