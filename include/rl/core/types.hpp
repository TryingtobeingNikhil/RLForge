#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rl::core {

// InfoMap carries auxiliary, algorithm-agnostic metadata returned alongside
// a step/reset (e.g. "episode_length", debugging fields a specific env wants
// to surface). It is intentionally a loose, small variant rather than a
// generic std::any bag -- this keeps it serializable and easy to log without
// pulling in reflection machinery we don't need yet.
using InfoValue = std::variant<bool, int64_t, double, std::string>;
using InfoMap = std::unordered_map<std::string, InfoValue>;

// A Value represents either a discrete index (e.g. "action 2 of 4") or a
// continuous vector (e.g. a 3-DoF torque command, or a raw pixel/feature
// observation). We use one variant type for both observations and actions
// so that a single Space hierarchy (Discrete / Box) can describe either one
// uniformly -- an Environment's action_space() and observation_space() both
// return `const Space&`, and both produce/consume `Value`.
//
// We deliberately do NOT introduce a tensor/ndarray type at this stage.
// Nothing downstream (this milestone) needs multi-dimensional buffers with
// strides -- that complexity is deferred to the neural network milestone,
// where we'll know what shape the abstraction actually needs to take.
using Value = std::variant<int64_t, std::vector<float>>;
using Observation = Value;
using Action = Value;

// Extracts the discrete index from a Value, or throws std::bad_variant_access
// with additional context if the Value does not hold an index. Centralizing
// this avoids every call site writing its own std::get<int64_t> and getting
// a cryptic std::bad_variant_access when an algorithm assumes the wrong kind
// of action space (e.g. DQN receiving a Box action).
int64_t as_index(const Value& value);

// Extracts the continuous vector from a Value, or throws with context if the
// Value does not hold one.
const std::vector<float>& as_vector(const Value& value);

// Result of calling Environment::step().
//
// `terminated` and `truncated` are kept separate (Gymnasium-style) rather
// than collapsed into a single `done` flag:
//   - terminated: the underlying MDP reached a true terminal state (e.g. the
//     agent fell off a cliff, or reached the goal). Value-based bootstrapping
//     must NOT continue past this transition -- there is no "next state".
//   - truncated: the episode was cut off for a reason outside the MDP itself
//     (e.g. a time limit). Bootstrapping SHOULD continue across this
//     transition, because the underlying MDP did not actually end.
// Conflating these two silently corrupts value function targets, and
// retrofitting the distinction after algorithms are written against a single
// `done` flag requires touching every return/advantage computation. We fix
// it here, once.
struct StepResult {
    Observation observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
    InfoMap info;
};

// Result of calling Environment::reset(). Deliberately has no reward/done
// fields -- there is no transition associated with a reset, only an initial
// observation.
struct ResetResult {
    Observation observation;
    InfoMap info;
};

// A per-sub-environment array of boolean flags (terminated/truncated across
// a batch of environments). Deliberately NOT std::vector<bool>: that
// specialization bit-packs its storage, and the resulting proxy references
// are not safe to write to concurrently even when two writers target
// different logical indices, since they may share the same underlying byte.
// A later multi-threaded rollout worker will write into these per-env
// slots from different threads -- using vector<bool> here would force a
// breaking change to this type at that point. std::vector<uint8_t> (treated
// as a bool) sidesteps the issue now.
using BoolArray = std::vector<uint8_t>;

// Result of calling VectorEnvironment::reset(). One entry per sub-env, in
// the same order the sub-envs were constructed in.
struct VectorResetResult {
    std::vector<Observation> observations;
    std::vector<InfoMap> infos;
};

// Result of calling VectorEnvironment::step(). One entry per sub-env.
//
// Auto-reset semantics: if terminated[i] or truncated[i] is true, the vector
// environment has already reset sub-env i internally, and observations[i]
// is that new episode's initial observation -- NOT the terminal observation
// of the episode that just ended. The terminal observation (needed to
// correctly bootstrap a value estimate for the transition that just
// occurred) is preserved in final_observations[i] instead. For any index
// where the episode did not end this step, final_observations[i] is empty.
//
// Silently discarding the terminal observation (returning only the
// post-reset one) is a common and easy-to-miss bug in home-grown vector env
// implementations: it makes the last transition of every episode look like
// it transitioned into the next episode's start state, which is simply
// wrong and will bias value estimates near episode boundaries.
struct VectorStepResult {
    std::vector<Observation> observations;
    std::vector<float> rewards;
    BoolArray terminated;
    BoolArray truncated;
    std::vector<InfoMap> infos;
    std::vector<std::optional<Observation>> final_observations;
};

} // namespace rl::core
