# Threaded Vector Environment (Milestone 9)

`rl::vector_envs::ThreadedVectorEnvironment` is a concurrent implementation
of the existing `VectorEnvironment` interface.

## Execution model

- Each factory is invoked inside its own persistent worker thread.
- The resulting environment is accessed only by that worker.
- A vector `reset()` or `step()` dispatches one command to every worker and
  waits for all results.
- Returned arrays retain factory order regardless of completion order.
- Workers remain alive across calls and are joined during destruction.

There is no thread creation in the rollout hot path.

## Contract compatibility

The implementation preserves the same behavior as
`SyncVectorEnvironment`:

- a non-empty, homogeneous environment set is required;
- base seed `s` produces per-environment seeds `s + i`;
- terminal and truncated environments auto-reset;
- the true final observation is returned in `final_observations`, while
  `observations` contains the reset state;
- worker exceptions are collected after every worker finishes and then
  rethrown in the caller.

Tensor gradient mode is thread-local as of this milestone, so an inference
thread's `no_grad()` scope cannot disable graph construction in a learner
thread.

## Usage

The constructor accepts the same `EnvFactory` type as the synchronous
implementation, making execution strategies interchangeable:

```cpp
std::vector<rl::vector_envs::EnvFactory> factories = /* ... */;
rl::vector_envs::ThreadedVectorEnvironment env(std::move(factories));
```
