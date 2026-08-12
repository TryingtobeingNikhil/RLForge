#pragma once

#include <memory>
#include <vector>

#include "rl/tensor/tensor.hpp"

namespace rl::optim {

// ---------------------------------------------------------------------------
// Optimizer — abstract base class for parameter update rules.
//
// Namespace/file layout follows the established convention:
//   rl::optim → include/rl/optim/ headers, src/optim/ implementations.
//
// Every Optimizer:
//   - Holds a vector of shared_ptr<Tensor> parameters. These are the same
//     shared_ptr objects returned by Module::parameters(), so the optimizer
//     has a stable reference to each parameter's actual data.
//   - Implements step() to update parameter values in-place based on their
//     current .grad() buffers. Each step() call increments the Storage
//     version counter of every mutated parameter (via data_mutable()), which
//     is what makes the Milestone 6 version guard effective: any stale
//     autograd graph from the forward pass that ran before this step() will
//     throw std::runtime_error if backward() is called again on it.
//   - Implements zero_grad() to clear all parameter grad buffers before the
//     next forward pass.
// ---------------------------------------------------------------------------
class Optimizer {
public:
    explicit Optimizer(std::vector<std::shared_ptr<rl::tensor::Tensor>> params);

    virtual ~Optimizer() = default;

    // Apply one parameter update step using currently accumulated gradients.
    // Mutates parameter data in-place (increments Storage version counter).
    virtual void step() = 0;

    // Zero out the grad buffer of every registered parameter.
    // Call this before each forward pass to prevent gradient accumulation
    // across iterations (backward() does not auto-zero).
    void zero_grad();

protected:
    std::vector<std::shared_ptr<rl::tensor::Tensor>> params_;
};

} // namespace rl::optim
