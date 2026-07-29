# Environment API

This document specifies the contract every environment in `rl-lib` must
satisfy. If you are implementing a new environment, read this instead of
inferring behavior from `GridWorld` — `GridWorld` is a reference
implementation, not the spec.

## Core types (`rl/core/types.hpp`)

- `Value = std::variant<int64_t, std::vector<float>>` — either a discrete
  index or a continuous vector. Used for both observations and actions.
- `StepResult { observation, reward, terminated, truncated, info }`
- `ResetResult { observation, info }`

### `terminated` vs `truncated`

- `terminated = true` means the underlying MDP reached a genuine terminal
  state (e.g. goal reached, agent died). **There is no valid next state.**
  Value-based algorithms must not bootstrap across this transition.
- `truncated = true` means the episode was cut off for a reason external to
  the MDP (typically a time limit). The MDP did not actually end — a
  well-behaved algorithm may still bootstrap the value of the next state
  across this transition, since it exists.
- These are never both `true` on the same step. If the environment reaches
  a genuine terminal state, report only `terminated`, even if it also
  happens to coincide with the time limit.

## Space (`rl/core/space.hpp`)

`Space` describes the shape/bounds of a set of valid `Value`s. It exposes:

- `sample(std::mt19937& rng) const -> Value` — draws a random valid value.
  The RNG is **always** supplied by the caller. A `Space` must never own or
  seed its own RNG; doing so breaks reproducibility, because seeding an
  `Environment` would then not deterministically control every random draw
  made against its spaces.
- `contains(const Value&) const -> bool` — validity check.
- `describe() const -> std::string` — human-readable, for logs/errors only.

Two concrete spaces exist: `Discrete(n)` (valid indices `0..n-1`) and
`Box(low, high)` (continuous, axis-aligned, elementwise bounded).

## Environment (`rl/core/environment.hpp`)

```cpp
class Environment {
public:
    virtual ~Environment() = default;
    virtual ResetResult reset(std::optional<uint64_t> seed = std::nullopt) = 0;
    virtual StepResult step(const Action& action) = 0;
    virtual const Space& observation_space() const = 0;
    virtual const Space& action_space() const = 0;
    virtual std::string name() const = 0;
};
```

### Contract every implementation must satisfy

1. **`step()` before the first `reset()` is a programmer error** and must
   raise an exception, not silently proceed. If you inherit `EnvironmentBase`
   (recommended — see below), this is enforced for you.
2. **`reset(seed)` must be deterministic.** Calling `reset(42)` twice, then
   driving both instances with the same action sequence, must produce
   identical `StepResult`s (same observation, reward, terminated, truncated)
   at every step. If your environment has no internal randomness, this is
   automatically satisfied. If it does (e.g. a "slip" chance, procedurally
   generated layouts), that randomness **must** be drawn from an RNG that
   `reset(seed)` reseeds — never from `std::rand()`, `time(nullptr)`, or a
   static/global RNG.
3. **Observations returned by `reset()`/`step()` must satisfy
   `observation_space().contains(observation)`.**
4. **Actions passed to `step()` should be validated against
   `action_space()`.** Reject invalid actions with a thrown exception rather
   than silently clamping or ignoring them — silent clamping hides bugs in
   whatever produced the action (a policy, a test, a human).

### `EnvironmentBase`: the recommended base class

Rather than implementing the "reset before step" guard in every
environment, inherit `EnvironmentBase` and override the protected
`reset_impl` / `step_impl` instead of `reset` / `step` directly:

```cpp
class MyEnv final : public rl::core::EnvironmentBase {
protected:
    rl::core::ResetResult reset_impl(std::optional<uint64_t> seed) override;
    rl::core::StepResult step_impl(const rl::core::Action& action) override;
    // ... observation_space(), action_space(), name() as usual
};
```

`EnvironmentBase::reset()`/`step()` are `final` and implement the
reset-before-step guard once; your `_impl` overrides never need to think
about it.

## Testing a new environment

Every environment's test file should call the shared contract suite in
`tests/contract/environment_contract.hpp` against a factory that constructs
a fresh instance, in addition to any environment-specific tests:

```cpp
TEST_CASE("MyEnv satisfies the Environment contract") {
    rl::testing::run_environment_contract_tests(
        []() -> std::unique_ptr<rl::core::Environment> {
            return std::make_unique<MyEnv>();
        });
}
```

This is not optional boilerplate — it is what catches regressions like
"forgot to reseed the RNG in reset()" before they surface as a
non-reproducible experiment three milestones from now.
