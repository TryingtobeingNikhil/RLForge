#pragma once

#include <cstdint>
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

} // namespace rl::core
