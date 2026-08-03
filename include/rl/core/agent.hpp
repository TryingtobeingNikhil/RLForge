#pragma once

#include <string>
#include <vector>

#include "rl/core/transition.hpp"
#include "rl/core/types.hpp"

namespace rl::core {

// Training/update metrics an Agent reports after a learning step (e.g.
// {"loss": 0.42, "td_error": 0.03}), for logging. Reuses InfoMap's variant
// shape rather than introducing a near-duplicate type -- the vocabulary
// differs (env metadata vs. training metrics) but the representation need
// not.
using Metrics = InfoMap;

// Agent is the abstraction every RL algorithm in this library implements --
// tabular Q-Learning first, DQN/PPO/SAC later, all behind this same
// interface, so Trainer (rl/core/trainer.hpp) never needs to know which
// kind of algorithm it's driving.
//
// The central design problem this interface solves: different algorithms
// learn on fundamentally different cadences.
//   - Tabular Q-Learning / DQN: learn from a transition (or a batch sampled
//     from a replay buffer) every step, or every few steps once a buffer is
//     "warm" (has enough data).
//   - PPO: accumulates a full on-policy rollout of fixed length, then runs
//     several epochs of updates over it, then discards it and starts over.
// Forcing one of these shapes onto the other -- e.g. requiring every
// algorithm to update once per environment step -- makes one of them
// awkward. Instead, four hooks are split apart so each algorithm decides
// its own cadence, while the calling code (Trainer) stays identical
// regardless of which algorithm it's driving:
//
//   act()                -- select actions for a batch of observations.
//   observe_transitions() -- ingest a batch of just-experienced transitions.
//                            Pure data intake: implementations should NOT
//                            perform a learning update inside this call, so
//                            that "collecting experience" and "learning
//                            from experience" stay decoupled -- particularly
//                            important once rollout collection becomes
//                            multi-threaded and these two responsibilities
//                            may run on different threads entirely.
//   should_update()       -- is the agent ready to learn right now? Tabular
//                            Q-Learning: yes, whenever there's a pending
//                            transition. DQN: yes, every train_freq steps,
//                            once the replay buffer has enough data. PPO:
//                            yes, once the on-policy rollout buffer is full.
//   update()              -- perform exactly one learning step and report
//                            metrics about it.
//
// act() takes and returns *batches* (one entry per sub-env of whatever
// VectorEnvironment the Trainer is driving), not a single observation/
// action. This matters for Deep RL specifically: a neural-network-backed
// agent should run one batched forward pass across all sub-envs, not N
// separate single-observation forward passes -- the interface needs to
// support that efficiently from the start, or every future algorithm pays
// for a shape mismatch fixed here.
class Agent {
public:
    virtual ~Agent() = default;

    // Selects one action per observation. `explore` distinguishes
    // training-time behavior (e.g. epsilon-greedy, or sampling from a
    // stochastic policy) from evaluation-time behavior (the agent's best
    // current estimate -- greedy/deterministic). Trainer calls this with
    // explore=true during train() and explore=false during evaluate().
    virtual std::vector<Action> act(const std::vector<Observation>& observations,
                                     bool explore) = 0;

    // Ingests one transition per sub-env, just experienced this step. Must
    // not perform a learning update -- only record/buffer data. What
    // "record" means is entirely up to the agent: an immediate online
    // update's *inputs* can be buffered here and consumed by update() next
    // (tabular Q-Learning), pushed into a replay buffer for later off-policy
    // sampling (DQN, SAC), or appended to an on-policy rollout buffer (PPO).
    virtual void observe_transitions(const std::vector<Transition>& transitions) = 0;

    // Whether update() should be called right now. Trainer checks this
    // after every observe_transitions() call and calls update() only if it
    // returns true -- this is the mechanism that lets algorithms with
    // entirely different learning cadences share one Trainer loop.
    virtual bool should_update() const = 0;

    // Performs exactly one learning update (whatever that means for this
    // algorithm) and returns metrics describing it, for logging.
    virtual Metrics update() = 0;

    // Human-readable identifier, for logging.
    virtual std::string name() const = 0;
};

} // namespace rl::core
