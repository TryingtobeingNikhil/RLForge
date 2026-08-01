#pragma once

#include <vector>

#include "rl/core/types.hpp"

namespace rl::core {

// A single (s, a, r, s', terminated, truncated) experience tuple -- the
// fundamental unit a ReplayBuffer stores and samples.
//
// next_observation must be the TRUE next observation of the underlying MDP,
// never a post-auto-reset observation from a VectorEnvironment. See
// make_transitions() below, which resolves this correctly when building
// Transitions from a VectorStepResult.
struct Transition {
    Observation observation;
    Action action;
    float reward = 0.0f;
    Observation next_observation;
    bool terminated = false;
    bool truncated = false;
};

// A sampled batch of transitions, as parallel arrays -- the same columnar
// shape as VectorStepResult, for consistency across the library and so a
// future flattening into contiguous tensors has one shape to convert
// instead of two independently-evolving ones.
struct TransitionBatch {
    std::vector<Observation> observations;
    std::vector<Action> actions;
    std::vector<float> rewards;
    std::vector<Observation> next_observations;
    BoolArray terminated;
    BoolArray truncated;
};

// Builds a Transition from a single Environment's StepResult. A plain
// Environment never auto-resets mid-step (see rl/core/environment.hpp), so
// step_result.observation is always the genuine next observation -- there
// is no ambiguity to resolve here, unlike the vectorized case below.
Transition make_transition(Observation observation, Action action,
                            const StepResult& step_result);

// Builds one Transition per sub-env from a VectorEnvironment's
// VectorStepResult, given the observations that preceded this step and the
// actions taken this step. For sub-env i, uses final_observations[i] (the
// true terminal observation) instead of observations[i] (the
// post-auto-reset observation) whenever sub-env i's episode ended this step
// -- this is exactly the distinction VectorStepResult exists to preserve
// (see docs/vector_environment_api.md). Using observations[i]
// unconditionally would silently corrupt every transition at an episode
// boundary, making the terminal transition look like it led into the next
// episode's start state rather than ending the MDP.
//
// `observations` and `actions` must each have the same size as
// step_result.observations; throws std::invalid_argument otherwise.
std::vector<Transition> make_transitions(
    const std::vector<Observation>& observations,
    const std::vector<Action>& actions, const VectorStepResult& step_result);

} // namespace rl::core
