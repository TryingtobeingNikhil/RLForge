#include "rl/core/environment.hpp"

#include <stdexcept>

namespace rl::core {

ResetResult EnvironmentBase::reset(std::optional<uint64_t> seed) {
    has_been_reset_ = true;
    return reset_impl(seed);
}

StepResult EnvironmentBase::step(const Action& action) {
    if (!has_been_reset_) {
        throw std::logic_error(
            "Environment::step() called before reset(). Every environment "
            "must be reset at least once (to establish an initial state and, "
            "if applicable, seed its RNG) before step() can be called.");
    }
    return step_impl(action);
}

} // namespace rl::core
