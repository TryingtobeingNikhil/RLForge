# Agent & Trainer API

This covers Milestone 4. Read the earlier docs first — this milestone's
entire point is that nothing here required changing `Environment`,
`VectorEnvironment`, or `ReplayBuffer`; it only adds new abstractions on top
of them.

## The problem this milestone solves

Every RL algorithm needs to (1) pick actions and (2) learn from experience.
But *when* an algorithm learns varies enormously:

| Algorithm | Learns from | Cadence |
|---|---|---|
| Tabular Q-Learning | the transition(s) just observed | every step |
| DQN | a batch sampled from a replay buffer | every N steps, once the buffer is warm |
| PPO | a full on-policy rollout | once per rollout, then discards it |
| SAC | a batch sampled from a replay buffer | every N steps, once the buffer is warm |

A `Trainer` that only knows how to call `update()` once per environment
step would force PPO into an awkward shape. A `Trainer` that only knows how
to accumulate a full rollout before updating would make Q-Learning learn
far too slowly. Neither should be baked into the training loop.

## `Agent` (`rl/core/agent.hpp`)

```cpp
class Agent {
public:
    virtual std::vector<Action> act(const std::vector<Observation>& observations, bool explore) = 0;
    virtual void observe_transitions(const std::vector<Transition>& transitions) = 0;
    virtual bool should_update() const = 0;
    virtual Metrics update() = 0;
    virtual std::string name() const = 0;
};
```

Four hooks, deliberately split:

- **`act()`** is batched — one call per `Trainer` step, one observation in
  and one action out per sub-env. This is a Deep-RL-motivated decision made
  now: a neural-network agent needs one batched forward pass across all
  sub-envs, not N separate single-observation passes. Baking a
  single-observation `act()` into the interface now would mean every future
  algorithm pays for a shape mismatch introduced here.
- **`observe_transitions()`** only ingests data — it must never perform a
  learning update itself. This keeps "collecting experience" and "learning
  from experience" decoupled, which matters most once rollout collection
  becomes multi-threaded (a later milestone): collection and learning may
  end up running on different threads entirely, and that split is much
  easier if it already exists here.
- **`should_update()`** is the mechanism that lets every algorithm above
  share one `Trainer` loop. `Trainer` never asks "is this a step-based or
  rollout-based algorithm" — it just calls `should_update()` after every
  `observe_transitions()` and calls `update()` if true. Tabular Q-Learning's
  implementation is `return !pending_transitions_.empty();`; a future PPO
  agent's would be `return rollout_buffer_.size() >= rollout_length_;`.
- **`update()`** performs one learning step and returns `Metrics` (an alias
  for `InfoMap`) describing it — e.g. `{"td_error": ..., "epsilon": ...}` —
  for whatever logging arrives later. Not itself a logging framework.

## `Trainer` (`rl/core/trainer.hpp`)

```cpp
class Trainer {
public:
    Trainer(VectorEnvironment& train_env, Environment& eval_env, Agent& agent);
    TrainingResult train(size_t num_steps, std::optional<uint64_t> seed = std::nullopt);
    EvaluationResult evaluate(size_t num_episodes, std::optional<uint64_t> seed = std::nullopt,
                               size_t max_steps_per_episode = 100000);
};
```

Two design choices worth calling out:

1. **Training is driven through `VectorEnvironment`, even with
   `num_envs() == 1`.** There is no separate single-env training code path.
   This means `Trainer` never needs to change when parallel rollout
   collection or a many-env DQN/PPO setup arrives — it already only ever
   talks to the `VectorEnvironment` interface. (Verified directly: a test
   constructs a `SyncVectorEnvironment` with exactly one sub-env and runs
   the same `Trainer` code against it, unmodified.)
2. **Evaluation uses a separate, non-vectorized `Environment`, not the
   training env.** Stepping the training env for evaluation purposes would
   disturb whichever in-flight episode is mid-collection for training.
   `evaluate()` also never calls `observe_transitions()`/`update()` — it
   only reads the agent's current policy via `act(..., explore=false)`,
   never mutates agent state.

`Trainer::train()`'s loop, per step:

```cpp
actions = agent.act(current_observations, /*explore=*/true);
step_result = train_env.step(actions);
transitions = make_transitions(current_observations, actions, step_result); // Milestone 3's bridge
agent.observe_transitions(transitions);
if (agent.should_update()) {
    metrics = agent.update();
}
current_observations = step_result.observations;
```

Note `make_transitions()` — this is the same function from Milestone 3 that
correctly substitutes `final_observations[i]` for `observations[i]` at
episode boundaries. `Trainer` uses it rather than building `Transition`s by
hand, so that correctness doesn't have to be re-derived here.

## `TabularQLearningAgent` (`rl/agents/tabular_q_learning_agent.hpp`)

The first (and, for this milestone, only) concrete `Agent`. Classic
epsilon-greedy tabular Q-Learning:

- **Requires a `Discrete` action space** (checked at construction) — tabular
  Q-Learning is only defined over a finite action set.
- **Does not require a `Discrete` observation space.** The Q-table is keyed
  by `(Observation, action)` pairs, hashed by their actual contents but
  compared by full equality on lookup (so a hash collision can never
  silently conflate two different states' values). This lets it operate
  directly on `GridWorld`'s existing `Box(x, y)` observation without
  requiring Milestone 1 to change how `GridWorld` represents state.
- **TD target correctly distinguishes `terminated` from `truncated`** —
  the entire reason that split was built into `StepResult` back in
  Milestone 1:

  ```cpp
  next_value = transition.terminated ? 0.0f : best_action_and_value(transition.next_observation).second;
  td_target = transition.reward + discount_factor * next_value;
  ```

  A transition that ended the episode via a time limit (`truncated`) still
  bootstraps through the estimated value of the next state; one that
  reached a genuine terminal state (`terminated`) does not, regardless of
  what that next state's value happens to be. This is verified directly by
  two tests that seed a large, deliberately-wrong bootstrap value at a
  given state, then check it's included or excluded exactly as expected —
  not inferred from overall training performance, but checked arithmetically.

## What this milestone proved

`tests/test_trainer_grid_world.cpp` trains a `TabularQLearningAgent` against
a 4-way vectorized `GridWorld` for 20,000 steps and evaluates before/after,
with every seed fixed for full reproducibility:

- **Before training:** mean evaluation return is exactly `-100.0` — the
  agent's untrained greedy policy (ties broken toward action 0, "Up") walks
  into a wall and sits there for the full 100-step time limit, every
  episode, deterministically.
- **After training:** mean evaluation return is exactly `3.0` — the true
  optimal return on a 5×5 `GridWorld` with no slip (an 8-move shortest path:
  seven steps at −1, the eighth reaching the goal at +10 instead: 
  7×(−1) + 10 = 3), achieved on every evaluation episode.

This is the milestone's real claim: the `Environment → VectorEnvironment →
Agent → Trainer` pipeline, assembled with no algorithm-specific shortcuts
across four milestones, is sufficient on its own to learn a working policy.

## What's deliberately not built yet

- **No logging framework.** `TrainingResult`/`EvaluationResult` return raw
  per-episode series and per-update `Metrics`; printing, aggregating, or
  persisting them is left to the caller until the logging milestone.
- **No `Trainer` interface/base class.** Unlike `Environment`/
  `VectorEnvironment`, `Trainer` is a single concrete class for now. Its
  core loop is expected to be *extended* (e.g. splitting rollout collection
  out for multi-threading) rather than *swapped* the way concrete
  environments or storage backends are, so introducing an abstract
  `Trainer` interface now would be speculative. Worth revisiting once the
  multi-threaded rollout milestone is in front of us.
- **No `Agent` reset/seeding hook beyond construction.** Reasonable for a
  stateless-per-episode policy like tabular Q-Learning; a future recurrent
  policy may need an explicit "reset hidden state between episodes" hook
  that doesn't exist yet.
