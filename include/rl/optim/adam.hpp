#pragma once

#include <memory>
#include <vector>

#include "rl/optim/optimizer.hpp"
#include "rl/tensor/tensor.hpp"

namespace rl::optim {

// ---------------------------------------------------------------------------
// Adam — Adaptive Moment Estimation optimizer.
//
// Standard Adam update rule with bias correction (Kingma & Ba, 2015):
//
//   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]          // first moment
//   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2        // second moment
//   m_hat = m[i] / (1 - beta1^t)                          // bias-corrected
//   v_hat = v[i] / (1 - beta2^t)                          // bias-corrected
//   param[i] -= lr * m_hat / (sqrt(v_hat) + eps)
//
// where t is the step counter (incremented by each step() call).
//
// Per-parameter state keying: state is indexed by position in the params_
// vector (integer index). This is simpler and more correct than raw pointer
// keying because: (a) integer indexing needs no map/set overhead, (b) the
// params_ vector is fixed at optimizer construction time so indices are
// stable for the optimizer's lifetime, (c) there is no risk of aliasing
// different parameters with the same pointer value.
//
// In-place mutation: step() calls param.data_mutable() which bumps the
// Storage version counter. This ensures any stale backward graph built from
// a pre-step forward pass will throw std::runtime_error if called again.
// ---------------------------------------------------------------------------
class Adam : public Optimizer {
public:
    // Default hyperparameters match the values recommended in the original
    // Adam paper and adopted by PyTorch and most deep learning frameworks.
    explicit Adam(std::vector<std::shared_ptr<rl::tensor::Tensor>> params,
                  double lr    = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double eps   = 1e-8);

    void step() override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;

    // Global step counter for bias correction. Incremented once per step().
    int64_t t_ = 0;

    // Per-parameter first and second moment buffers, indexed by position in
    // params_. m_[i] and v_[i] have the same number of elements as params_[i].
    std::vector<std::vector<double>> m_;   // first moment (mean)
    std::vector<std::vector<double>> v_;   // second moment (uncentered variance)
};

} // namespace rl::optim
