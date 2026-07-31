#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "rl/core/vector_environment.hpp"

namespace rl::testing {

// Reusable contract tests for any VectorEnvironment implementation, mirroring
// run_environment_contract_tests for single environments. `factory` must
// return a freshly constructed, never-reset VectorEnvironment with at least
// 2 sub-envs each time it is called (2, so tests can distinguish "batched
// correctly" from "accidentally works for a single env").
void run_vector_environment_contract_tests(
    const std::function<std::unique_ptr<rl::core::VectorEnvironment>()>& factory);

} // namespace rl::testing
