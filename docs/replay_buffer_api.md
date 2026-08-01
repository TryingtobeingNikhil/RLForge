# Replay Buffer API

This covers the replay buffer introduced in Milestone 3. Read
`environment_api.md` and `vector_environment_api.md` first — the
terminated/truncated and final-observation handling in those documents is
exactly what this buffer depends on being correct.

## Design intent

The public API (`ReplayBuffer`) is meant to be stable for the life of the
project. The physical memory layout of stored transitions is not — it is
expected to change once the neural network milestone determines what
representation is actually worth optimizing for (e.g. flat, contiguous
per-field buffers instead of a `Transition` per entry). This document
describes where that boundary sits.

**`ReplayBuffer` never itself decides how a `Transition` is stored.** It
owns only:
- ring-buffer bookkeeping (current size, next write index, wraparound), and
- the sampling policy (currently: uniform at random, with replacement).

Everything about physical storage is delegated to a `TransitionStorage`
implementation, injected at construction. This is a Strategy pattern, not
an optimization deferred by convention — `ReplayBuffer`'s source has no
reference to any concrete storage type, so a future contiguous-buffer
backend is a new `TransitionStorage` implementation, not a change to
`ReplayBuffer` itself.

## `Transition` and `TransitionBatch` (`rl/core/transition.hpp`)

```cpp
struct Transition {
    Observation observation;
    Action action;
    float reward = 0.0f;
    Observation next_observation;
    bool terminated = false;
    bool truncated = false;
};
```

`next_observation` must be the environment's **true** next observation —
never a post-auto-reset observation from a `VectorEnvironment`. Two helpers
build `Transition`s correctly:

- `make_transition(observation, action, StepResult)` — for a single
  `Environment`. No ambiguity: a plain `Environment` never auto-resets
  mid-`step()`, so `step_result.observation` is always correct.
- `make_transitions(observations, actions, VectorStepResult)` — for a
  `VectorEnvironment`. For each sub-env `i`, uses `final_observations[i]`
  instead of `observations[i]` whenever that sub-env's episode ended this
  step. **Always use this helper rather than reading `VectorStepResult`
  fields directly when building transitions** — reading `observations[i]`
  unconditionally silently corrupts the transition at every episode
  boundary.

`TransitionBatch` is the columnar (struct-of-arrays) result of
`ReplayBuffer::sample()` — same shape as `VectorStepResult`, for
consistency and so a future flattening-to-tensors step has one shape to
convert, not two.

## `TransitionStorage` (`rl/core/replay_buffer.hpp`)

```cpp
class TransitionStorage {
public:
    virtual void set(size_t index, Transition transition) = 0;
    virtual Transition get(size_t index) const = 0;
    virtual size_t capacity() const = 0;
};
```

Implementations only need to answer "store/retrieve a `Transition` at a
given slot" — no ring-buffer logic, no sampling logic. `index` is always
valid (`< capacity()`) by the time `ReplayBuffer` calls in; a
`TransitionStorage` never needs to bounds-check against its own policy,
though implementations should still bounds-check defensively (see
`VectorTransitionStorage`, which throws `std::out_of_range` rather than
trusting the caller).

## `ReplayBuffer` (`rl/core/replay_buffer.hpp`)

```cpp
explicit ReplayBuffer(std::unique_ptr<TransitionStorage> storage);
void add(Transition transition);
TransitionBatch sample(size_t batch_size, std::mt19937& rng) const;
size_t size() const noexcept;
size_t capacity() const noexcept;
```

Capacity comes from `storage->capacity()` — there is deliberately no
separate capacity constructor argument, so the two can never disagree.

`sample()` takes the RNG by reference rather than owning one internally,
matching `Space::sample()`'s convention: an entire experiment's randomness
(environment stepping, network initialization, buffer sampling) should be
drivable from one seeded stream the training loop controls, so a full run
is reproducible end to end.

## `VectorTransitionStorage` (`rl/replay_buffers/vector_transition_storage.hpp`)

The reference `TransitionStorage` backend for this milestone: a plain
`std::vector<Transition>` sized to capacity at construction. Correct,
simple, and allocation-heavy — every continuous observation/action is its
own heap-allocated `vector<float>` inside the stored `Transition`. This is
the backend Milestone 3 ships with; it is expected to eventually be
replaced by a contiguous backend once the tensor/NN milestone settles on a
layout, without `ReplayBuffer`'s API changing.

It lives in `rl::replay_buffers`, not `rl::core` — the same reason
`GridWorld` isn't in `rl::core::envs` and `SyncVectorEnvironment` isn't in
`rl::core`: `core` should only ever depend on interfaces, never on one
specific concrete strategy among several. Depending on `rl_core` alone
should never pull in a specific storage backend, environment, or execution
strategy.

## Writing a new `TransitionStorage` backend

Implement `set`/`get`/`capacity`; `ReplayBuffer` needs nothing else. There
is a test in `tests/test_replay_buffer.cpp`
(`ReplayBuffer works against a TransitionStorage implementation other than
the bundled one`) that constructs a second, independent implementation and
verifies `ReplayBuffer` operates against it correctly — a template worth
copying when testing a new backend, since it verifies the abstraction is
actually honored rather than merely documented.
