#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "rl/core/space.hpp"
#include "rl/core/types.hpp"

namespace rl::core {

// Environment is the central extensibility point of the library: a pure
// abstract interface so that user code, third-party envs, and vectorized
// env wrappers can all hold `std::unique_ptr<Environment>` regardless of
// the concrete env type, without recompiling against it. This is the
// virtual-dispatch trade-off discussed for this milestone -- step() is
// called millions of times over a training run, but in practice env logic
// (physics, game state, I/O) dwarfs the cost of one virtual call. If a
// specific env's inner loop is later shown by profiling to be dominated by
// dispatch overhead, a template/CRTP fast path can be added underneath this
// interface without breaking it.
class Environment {
public:
    virtual ~Environment() = default;

    // Resets the environment to an initial state and returns the initial
    // observation. If `seed` is provided, the environment's internal RNG
    // (if any) must be reseeded such that a fixed action sequence played
    // after two resets with the same seed produces identical trajectories.
    // Must be called at least once before step().
    virtual ResetResult reset(std::optional<uint64_t> seed = std::nullopt) = 0;

    // Advances the environment by one transition given `action`.
    // Calling step() before the first reset() is a programmer error and
    // must not silently proceed (see EnvironmentBase for the enforced
    // guard) -- it should surface as an exception, not undefined behavior
    // or a silently-invalid observation, since this is the kind of bug
    // that is otherwise very hard to catch inside a multi-threaded rollout
    // worker.
    virtual StepResult step(const Action& action) = 0;

    // Describes the shape/bounds of observations this environment produces.
    virtual const Space& observation_space() const = 0;

    // Describes the shape/bounds of actions this environment accepts.
    virtual const Space& action_space() const = 0;

    // Human-readable identifier, used for logging, registries, and
    // multi-env bookkeeping later (e.g. vectorized envs reporting which
    // sub-env raised an error).
    virtual std::string name() const = 0;
};

// EnvironmentBase implements the "step() before reset()" contract exactly
// once, via the Template Method pattern: reset()/step() are non-virtual and
// final, and delegate to protected reset_impl()/step_impl() that concrete
// environments override instead. This means every concrete environment gets
// the guard for free and cannot accidentally forget it -- a mistake that
// would otherwise surface as a confusing bug deep inside a rollout loop
// rather than a clear exception at the call site.
//
// Environment itself remains a plain abstract interface (not all
// implementers are required to inherit EnvironmentBase), but it is the
// recommended base for any concrete environment in this library.
class EnvironmentBase : public Environment {
public:
    ResetResult reset(std::optional<uint64_t> seed = std::nullopt) final;
    StepResult step(const Action& action) final;

protected:
    virtual ResetResult reset_impl(std::optional<uint64_t> seed) = 0;
    virtual StepResult step_impl(const Action& action) = 0;

private:
    bool has_been_reset_ = false;
};

} // namespace rl::core
