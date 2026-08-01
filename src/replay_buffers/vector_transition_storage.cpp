#include "rl/replay_buffers/vector_transition_storage.hpp"

#include <stdexcept>
#include <utility>

namespace rl::replay_buffers {

VectorTransitionStorage::VectorTransitionStorage(size_t capacity)
    : storage_(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument(
            "VectorTransitionStorage capacity must be > 0");
    }
}

void VectorTransitionStorage::set(size_t index, rl::core::Transition transition) {
    if (index >= storage_.size()) {
        throw std::out_of_range(
            "VectorTransitionStorage::set() index out of range");
    }
    storage_[index] = std::move(transition);
}

rl::core::Transition VectorTransitionStorage::get(size_t index) const {
    if (index >= storage_.size()) {
        throw std::out_of_range(
            "VectorTransitionStorage::get() index out of range");
    }
    return storage_[index];
}

} // namespace rl::replay_buffers
