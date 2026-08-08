#include "rl/optim/optimizer.hpp"

namespace rl::optim {

void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p) { p->zero_grad(); }
    }
}

} // namespace rl::optim
