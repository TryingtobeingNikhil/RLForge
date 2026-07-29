#include "rl/core/types.hpp"

#include <stdexcept>

namespace rl::core {

int64_t as_index(const Value& value) {
    if (const auto* index = std::get_if<int64_t>(&value)) {
        return *index;
    }
    throw std::invalid_argument(
        "as_index() called on a Value holding a continuous vector, not a "
        "discrete index. This usually means an algorithm built for a "
        "Discrete space was given a Box action/observation, or vice versa.");
}

const std::vector<float>& as_vector(const Value& value) {
    if (const auto* vec = std::get_if<std::vector<float>>(&value)) {
        return *vec;
    }
    throw std::invalid_argument(
        "as_vector() called on a Value holding a discrete index, not a "
        "continuous vector. This usually means an algorithm built for a "
        "Box space was given a Discrete action/observation, or vice versa.");
}

} // namespace rl::core
