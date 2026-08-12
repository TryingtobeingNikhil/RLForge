# Proximal Policy Optimization (Milestone 8)

RLForge provides a discrete-action PPO agent through `rl::agents::PPOAgent`.
It implements the existing `rl::core::Agent` contract, so the unchanged
`Trainer` collects a fixed on-policy rollout and calls `update()` when the
rollout reaches `PPOConfig::rollout_steps`.

## Components

- `ActorCriticNetwork`: independent actor and critic MLPs. The actor emits
  unnormalized categorical logits; the critic emits one scalar per state.
- `RolloutBuffer`: stores one batched vector-environment step per row and
  preserves environment lanes for reverse-time advantage calculation.
- `compute_gae`: computes generalized advantage estimates over `[T,N]` data.
- `PPOAgent`: samples actions, records old log-probabilities and values, then
  runs shuffled minibatch updates over multiple epochs.

## Correct episode-boundary semantics

GAE treats the two ending flags differently:

- `terminated` sets the value bootstrap to zero and breaks the GAE trace.
- `truncated` keeps the value bootstrap because the MDP continues, but breaks
  the trace because the vector environment auto-resets that lane.

This prevents returns from leaking from a reset episode into the preceding
one while retaining the correct time-limit value target.

## Objective

For each minibatch, PPO minimizes the negative clipped surrogate objective,
adds a squared value error, and subtracts categorical entropy:

```text
ratio = exp(new_log_probability - old_log_probability)
policy_loss = -mean(min(ratio * advantage,
                        clamp(ratio, 1-epsilon, 1+epsilon) * advantage))
loss = policy_loss + value_coefficient * value_loss
                   - entropy_coefficient * entropy
```

Reported metrics include policy loss, value loss, entropy, approximate KL,
clip fraction, and the number of minibatch updates.

## Scope

This milestone supports `Box` vector observations and `Discrete` actions.
Continuous-action distributions, recurrent policies, observation
normalization, and checkpoint serialization are not part of this milestone.
