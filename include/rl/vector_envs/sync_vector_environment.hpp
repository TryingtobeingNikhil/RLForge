#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "rl/core/environment.hpp"
#include "rl/core/vector_environment.hpp"

namespace rl::vector_envs {

// A factory that constructs one sub-environment. SyncVectorEnvironment takes
// a vector of these rather than a vector of already-constructed
// Environments: a future async/subprocess vector environment must construct
// each sub-env *inside* its own worker thread or process rather than
// sharing one built in the caller's context, and accepting factories here
// means that later implementation is a drop-in alternative to this one
// rather than a different constructor signature.
using EnvFactory = std::function<std::unique_ptr<rl::core::Environment>()>;

// SyncVectorEnvironment steps its sub-environments sequentially, in a single
// thread, in the calling thread. It exists to establish the
// VectorEnvironment contract (batching, auto-reset, seed derivation) with
// the simplest possible execution strategy before a concurrent
// implementation is introduced in a later milestone.
class SyncVectorEnvironment final : public rl::core::VectorEnvironmentBase {
public:
    // `factories` must be non-empty. All constructed sub-envs must report
    // identical observation_space()/action_space() (checked via
    // rl::core::spaces_compatible); a mismatch throws std::invalid_argument.
    explicit SyncVectorEnvironment(std::vector<EnvFactory> factories);

    size_t num_envs() const override { return envs_.size(); }
    const rl::core::Space& observation_space() const override {
        return envs_.front()->observation_space();
    }
    const rl::core::Space& action_space() const override {
        return envs_.front()->action_space();
    }

protected:
    rl::core::VectorResetResult reset_impl(std::optional<uint64_t> seed) override;
    rl::core::VectorStepResult step_impl(
        const std::vector<rl::core::Action>& actions) override;

private:
    std::vector<std::unique_ptr<rl::core::Environment>> envs_;
};

} // namespace rl::vector_envs
