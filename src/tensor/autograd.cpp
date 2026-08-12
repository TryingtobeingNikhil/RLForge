#include "rl/tensor/autograd.hpp"

namespace rl::tensor {

namespace {
// Thread-local gradient mode keeps concurrent inference and learner scopes
// independent.
thread_local bool s_grad_enabled = true;
} // namespace

bool grad_mode_enabled() noexcept {
    return s_grad_enabled;
}

NoGradGuard::NoGradGuard() : saved_mode_(s_grad_enabled) {
    s_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    s_grad_enabled = saved_mode_;
}

NoGradGuard no_grad() {
    return NoGradGuard{};
}

} // namespace rl::tensor
