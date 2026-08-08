#pragma once

#include <memory>
#include <vector>

#include "rl/optim/optimizer.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::optim {

// ---------------------------------------------------------------------------
// SGD — Stochastic Gradient Descent with optional momentum.
//
// Update rule (plain SGD, momentum=0):
//   param -= lr * grad
//
// Update rule (with momentum > 0):
//   velocity[i] = momentum * velocity[i] + grad[i]
//   param[i]    -= lr * velocity[i]
//
// Momentum buffers are initialised to zero on first step() call and persist
// across steps. They are keyed by parameter index in the params_ vector.
//
// In-place mutation: step() calls param.data_mutable() which bumps the
// Storage version counter. This ensures any stale backward graph built from
// a pre-step forward pass will throw std::runtime_error if called again.
// ---------------------------------------------------------------------------
class SGD : public Optimizer {
public:
    // lr       — learning rate (must be > 0)
    // momentum — momentum coefficient (0 = plain SGD, typical values 0.9)
    explicit SGD(std::vector<std::shared_ptr<rl::tensor::Tensor>> params,
                 double lr,
                 double momentum = 0.0);

    void step() override;

private:
    double lr_;
    double momentum_;

    // Per-parameter velocity buffers for momentum.
    // velocity_[i] has the same shape as params_[i].
    // Empty if momentum_ == 0.0.
    std::vector<std::vector<double>> velocity_;
};

} // namespace rl::optim
