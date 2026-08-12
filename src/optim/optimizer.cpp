#include "rl/optim/optimizer.hpp"

#include <stdexcept>
#include <utility>

namespace rl::optim {

Optimizer::Optimizer(std::vector<std::shared_ptr<rl::tensor::Tensor>> params)
    : params_(std::move(params)) {
    for (const auto& param : params_) {
        if (!param) {
            throw std::invalid_argument(
                "Optimizer parameters must not contain null tensors");
        }
    }
}

void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p) { p->zero_grad(); }
    }
}

} // namespace rl::optim
