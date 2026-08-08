#include "rl/optim/sgd.hpp"

#include <stdexcept>

namespace rl::optim {

SGD::SGD(std::vector<std::shared_ptr<rl::tensor::Tensor>> params,
         double lr,
         double momentum)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum) {
    if (momentum_ > 0.0) {
        // Initialise velocity buffers to zero for each parameter.
        velocity_.reserve(params_.size());
        for (const auto& p : params_) {
            velocity_.emplace_back(static_cast<size_t>(p->numel()), 0.0);
        }
    }
}

void SGD::step() {
    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& p = *params_[idx];
        if (!p.requires_grad() || p.grad() == nullptr) { continue; }

        const auto& g = p.grad()->data();
        // data_mutable() bumps the Storage version counter — this is what
        // makes the version guard effective after optimizer steps.
        auto& data = p.data_mutable();

        if (momentum_ == 0.0) {
            // Plain SGD: param -= lr * grad
            for (size_t i = 0; i < static_cast<size_t>(p.numel()); ++i) {
                data[i] -= lr_ * g[i];
            }
        } else {
            // SGD with momentum:
            //   velocity = momentum * velocity + grad
            //   param   -= lr * velocity
            auto& vel = velocity_[idx];
            for (size_t i = 0; i < static_cast<size_t>(p.numel()); ++i) {
                vel[i]  = momentum_ * vel[i] + g[i];
                data[i] -= lr_ * vel[i];
            }
        }
    }
}

} // namespace rl::optim
