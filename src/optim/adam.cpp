#include "rl/optim/adam.hpp"

#include <cmath>

namespace rl::optim {

Adam::Adam(std::vector<std::shared_ptr<rl::tensor::Tensor>> params,
           double lr,
           double beta1,
           double beta2,
           double eps)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    // Initialise first and second moment buffers to zero for each parameter.
    m_.reserve(params_.size());
    v_.reserve(params_.size());
    for (const auto& p : params_) {
        m_.emplace_back(static_cast<size_t>(p->numel()), 0.0);
        v_.emplace_back(static_cast<size_t>(p->numel()), 0.0);
    }
}

void Adam::step() {
    ++t_;  // increment step counter (bias correction denominator)

    // Bias correction factors (precomputed once per step).
    const double bc1 = 1.0 - std::pow(beta1_, static_cast<double>(t_));
    const double bc2 = 1.0 - std::pow(beta2_, static_cast<double>(t_));

    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& p = *params_[idx];
        if (!p.requires_grad() || p.grad() == nullptr) { continue; }

        const auto& g    = p.grad()->data();
        auto& m          = m_[idx];
        auto& v          = v_[idx];
        // data_mutable() bumps the Storage version counter — this is what
        // makes the version guard effective after optimizer steps.
        auto& data       = p.data_mutable();
        const size_t n   = static_cast<size_t>(p.numel());

        for (size_t i = 0; i < n; ++i) {
            // Update biased first and second moment estimates.
            m[i] = beta1_ * m[i] + (1.0 - beta1_) * g[i];
            v[i] = beta2_ * v[i] + (1.0 - beta2_) * g[i] * g[i];

            // Bias-corrected estimates.
            const double m_hat = m[i] / bc1;
            const double v_hat = v[i] / bc2;

            // Parameter update.
            data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}

} // namespace rl::optim
