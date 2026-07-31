#pragma once

#include <cstddef>
#include <optional>

#include "rl/core/space.hpp"
#include "rl/core/types.hpp"

namespace rl::core {

// VectorEnvironment is the batched counterpart of Environment: it steps
// `num_envs()` sub-environments together and returns their results as
// parallel arrays (see VectorStepResult / VectorResetResult). It is an
// abstract interface for the same reason Environment is: this milestone
// implements only a synchronous (single-threaded) vectorization strategy
// (SyncVectorEnvironment), but a later multi-threaded or subprocess-based
// implementation must be swappable underneath rollout-collection code
// without that code changing at all.
//
// All sub-envs of a VectorEnvironment are required to share the same
// observation_space() and action_space() -- batched algorithms downstream
// (replay buffers, batched network inference) assume a uniform shape across
// the batch dimension. A VectorEnvironment does not support heterogeneous
// sub-envs; if you need that, you want a plain std::vector<unique_ptr
// <Environment>> stepped individually, not this interface.
class VectorEnvironment {
public:
    virtual ~VectorEnvironment() = default;

    // Resets every sub-environment. If `seed` is provided, sub-env i is
    // reset with derived seed `*seed + i` (not the same seed for every
    // sub-env) -- a shared seed would make every sub-env's trajectory
    // correlated (often identical), defeating the purpose of vectorizing in
    // the first place. The whole batch remains reproducible from one seed.
    virtual VectorResetResult reset(std::optional<uint64_t> seed = std::nullopt) = 0;

    // Steps every sub-environment with its corresponding action.
    // `actions.size()` must equal `num_envs()`.
    //
    // Any sub-env whose episode ends this step (terminated or truncated) is
    // automatically reset before this call returns; see VectorStepResult's
    // documentation for how the pre-reset terminal observation is
    // preserved.
    virtual VectorStepResult step(const std::vector<Action>& actions) = 0;

    virtual size_t num_envs() const = 0;

    // Shared across all sub-envs (validated at construction time).
    virtual const Space& observation_space() const = 0;
    virtual const Space& action_space() const = 0;
};

// VectorEnvironmentBase implements the reset-before-step guard exactly once,
// mirroring EnvironmentBase's Template Method pattern in environment.hpp.
// Concrete vector environments override reset_impl/step_impl instead of
// reset/step directly.
class VectorEnvironmentBase : public VectorEnvironment {
public:
    VectorResetResult reset(std::optional<uint64_t> seed = std::nullopt) final;
    VectorStepResult step(const std::vector<Action>& actions) final;

protected:
    virtual VectorResetResult reset_impl(std::optional<uint64_t> seed) = 0;
    virtual VectorStepResult step_impl(const std::vector<Action>& actions) = 0;

private:
    bool has_been_reset_ = false;
};

} // namespace rl::core
