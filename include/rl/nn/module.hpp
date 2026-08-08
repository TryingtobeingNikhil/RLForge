#pragma once

#include <memory>
#include <vector>

#include "rl/tensor/tensor.hpp"

namespace rl::nn {

// ---------------------------------------------------------------------------
// Module — abstract base class for all neural network layers.
//
// Design follows the established rl::tensor namespace/file layout convention:
//   rl::nn namespace → include/rl/nn/ headers, src/nn/ implementations.
//
// Every Module subclass must:
//   1. Implement forward() to compute the layer's output given an input Tensor.
//   2. Implement parameters() to return all learnable parameter tensors as
//      shared_ptr<Tensor>. Shared pointers are required because optimizers
//      hold references to the same parameter objects across multiple forward
//      calls — they must remain alive and their identity must be stable.
//
// Usage pattern:
//   rl::nn::Linear layer(4, 8);
//   auto out = layer.forward(x);           // forward pass
//   loss.backward();                       // backward pass
//   optimizer.step();                      // update parameters in-place
//   optimizer.zero_grad();                 // clear gradients for next step
// ---------------------------------------------------------------------------
class Module {
public:
    virtual ~Module() = default;

    // Compute the forward pass. Input and output shapes are layer-defined.
    virtual rl::tensor::Tensor forward(const rl::tensor::Tensor& input) = 0;

    // Return all learnable parameters. Each element is a shared_ptr<Tensor>
    // with requires_grad()==true. The returned vector is stable across calls:
    // the same shared_ptr objects are returned every time, allowing optimizers
    // to safely hold references to parameters by index.
    virtual std::vector<std::shared_ptr<rl::tensor::Tensor>> parameters() const = 0;
};

} // namespace rl::nn
