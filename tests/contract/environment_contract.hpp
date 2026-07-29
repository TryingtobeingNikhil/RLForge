#pragma once

#include <functional>
#include <memory>

#include "rl/core/environment.hpp"

namespace rl::testing {

// A reusable contract test suite: any Environment implementation, present or
// future, should pass this. Rather than re-writing "does step() reject
// calls before reset()" for every new environment we add later (vectorized
// wrappers, physics envs, ...), each new environment's test file calls this
// once against a factory that constructs it. This is the same reasoning as
// EnvironmentBase itself: enforce the contract in one place, reuse it
// everywhere.
//
// `factory` must return a freshly constructed, never-reset Environment each
// time it is called.
void run_environment_contract_tests(
    const std::function<std::unique_ptr<rl::core::Environment>()>& factory);

} // namespace rl::testing
