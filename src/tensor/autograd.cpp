#include "rl/tensor/autograd.hpp"

namespace rl::tensor {

namespace {
// Single global flag. Not thread_local — single-threaded training loop is
// assumed for this milestone. Thread safety revisit is deferred.
bool s_grad_enabled = true;
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
