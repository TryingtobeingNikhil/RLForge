#include "rl/core/vector_environment.hpp"

#include <stdexcept>

namespace rl::core {

VectorResetResult VectorEnvironmentBase::reset(std::optional<uint64_t> seed) {
    has_been_reset_ = true;
    return reset_impl(seed);
}

VectorStepResult VectorEnvironmentBase::step(const std::vector<Action>& actions) {
    if (!has_been_reset_) {
        throw std::logic_error(
            "VectorEnvironment::step() called before reset(). Every vector "
            "environment must be reset at least once before step() can be "
            "called.");
    }
    return step_impl(actions);
}

} // namespace rl::core
