# Vector Environment API

This extends `docs/environment_api.md` — read that first. This document
covers the batched interface introduced in Milestone 2.

## Why this exists

Training almost always collects rollouts from many environment instances at
once (better GPU utilization for batched inference, more decorrelated data
per update). `VectorEnvironment` is the interface that batches a set of
sub-environments behind one API, so algorithm code never needs to know
whether it's talking to 1 environment or 256, or whether they're stepped
sequentially in one thread or in parallel across a thread pool.

## Scope of this milestone

`SyncVectorEnvironment` steps its sub-environments **sequentially, in the
calling thread**. This is deliberate: multi-threaded / subprocess-based
rollout collection is a separate, later milestone. Building it now would mix
concurrency concerns into the same change that establishes the batching
contract itself — get the contract right first, parallelize the execution
strategy underneath it later without changing the interface.

## `VectorEnvironment` (`rl/core/vector_environment.hpp`)

```cpp
class VectorEnvironment {
public:
    virtual VectorResetResult reset(std::optional<uint64_t> seed = std::nullopt) = 0;
    virtual VectorStepResult step(const std::vector<Action>& actions) = 0;
    virtual size_t num_envs() const = 0;
    virtual const Space& observation_space() const = 0;
    virtual const Space& action_space() const = 0;
};
```

### Constraints every implementation must satisfy

1. **All sub-envs share one `observation_space()` and `action_space()`.**
   Validate this at construction (see `rl::core::spaces_compatible`) and
   fail loudly, not at some later batched-inference call site.
2. **`reset(seed)` derives a distinct seed per sub-env** (`*seed + i`), not
   one shared seed across all of them. A shared seed correlates every
   sub-env's trajectory — often making them identical — which defeats the
   purpose of vectorizing at all. The whole batch is still fully
   reproducible from the one base seed.
3. **Auto-reset, with the terminal observation preserved.** When sub-env `i`
   terminates or truncates during `step()`, that sub-env is reset
   immediately, and `observations[i]` in the returned `VectorStepResult` is
   already the *new* episode's initial observation. The observation that
   actually ended the previous episode is returned separately, in
   `final_observations[i]` (`std::optional<Observation>`, populated **iff**
   `terminated[i] || truncated[i]`).

   This split exists because algorithms need both things for different
   purposes: the transition that just happened (which needs the *true* next
   observation for a correct value target) and the environment ready to
   accept the next action (which needs to already be reset). Returning only
   the post-reset observation — a common shortcut — silently corrupts every
   value estimate at an episode boundary, since it looks like the episode
   transitioned into the next episode's start state.
4. **No `std::vector<bool>`.** `terminated`/`truncated` are
   `rl::core::BoolArray` (`= std::vector<uint8_t>`), specifically so a
   future multi-threaded rollout worker can write into different indices
   from different threads without fighting `vector<bool>`'s bit-packed,
   non-thread-safe proxy references.

## `SyncVectorEnvironment` (`rl/vector_envs/sync_vector_environment.hpp`)

Constructed from a `std::vector<EnvFactory>`
(`EnvFactory = std::function<std::unique_ptr<Environment>()>`), **not** a
vector of already-constructed environments:

```cpp
std::vector<EnvFactory> factories;
for (int i = 0; i < 8; ++i) {
    factories.push_back([]() -> std::unique_ptr<rl::core::Environment> {
        return std::make_unique<GridWorld>();
    });
}
SyncVectorEnvironment vec_env(std::move(factories));
```

Factories, not instances, because a future asynchronous implementation must
construct each sub-env inside its own worker thread or process — accepting
factories now means that implementation is a drop-in alternative to this
one, not a different constructor signature.

## Testing a new `VectorEnvironment` implementation

As with single environments, call the shared suite in
`tests/contract/vector_environment_contract.hpp` against a factory
producing at least 2 sub-envs (so batching bugs — e.g. accidentally
returning env 0's result for every slot — are actually detectable):

```cpp
TEST_CASE("MyVectorEnv satisfies the VectorEnvironment contract") {
    rl::testing::run_vector_environment_contract_tests(
        []() -> std::unique_ptr<rl::core::VectorEnvironment> {
            return std::make_unique<MyVectorEnv>(/* ... */);
        });
}
```
