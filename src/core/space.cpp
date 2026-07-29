#include "rl/core/space.hpp"

#include <stdexcept>

namespace rl::core {

// ---------------------------------------------------------------------------
// Discrete
// ---------------------------------------------------------------------------

Discrete::Discrete(int64_t n) : n_(n) {
    if (n_ <= 0) {
        throw std::invalid_argument("Discrete space must have n > 0, got " +
                                     std::to_string(n_));
    }
}

Value Discrete::sample(std::mt19937& rng) const {
    std::uniform_int_distribution<int64_t> dist(0, n_ - 1);
    return Value{dist(rng)};
}

bool Discrete::contains(const Value& value) const {
    const auto* index = std::get_if<int64_t>(&value);
    if (index == nullptr) {
        return false;
    }
    return *index >= 0 && *index < n_;
}

std::string Discrete::describe() const {
    return "Discrete(" + std::to_string(n_) + ")";
}

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------

Box::Box(std::vector<float> low, std::vector<float> high)
    : low_(std::move(low)), high_(std::move(high)) {
    if (low_.size() != high_.size()) {
        throw std::invalid_argument(
            "Box low/high must have the same size, got low.size()=" +
            std::to_string(low_.size()) +
            " high.size()=" + std::to_string(high_.size()));
    }
    if (low_.empty()) {
        throw std::invalid_argument("Box space must have dim > 0");
    }
    for (size_t i = 0; i < low_.size(); ++i) {
        if (low_[i] > high_[i]) {
            throw std::invalid_argument(
                "Box low[" + std::to_string(i) + "]=" +
                std::to_string(low_[i]) + " exceeds high[" +
                std::to_string(i) + "]=" + std::to_string(high_[i]));
        }
    }
}

Value Box::sample(std::mt19937& rng) const {
    std::vector<float> result(low_.size());
    for (size_t i = 0; i < low_.size(); ++i) {
        std::uniform_real_distribution<float> dist(low_[i], high_[i]);
        result[i] = dist(rng);
    }
    return Value{std::move(result)};
}

bool Box::contains(const Value& value) const {
    const auto* vec = std::get_if<std::vector<float>>(&value);
    if (vec == nullptr) {
        return false;
    }
    if (vec->size() != low_.size()) {
        return false;
    }
    for (size_t i = 0; i < vec->size(); ++i) {
        if ((*vec)[i] < low_[i] || (*vec)[i] > high_[i]) {
            return false;
        }
    }
    return true;
}

std::string Box::describe() const {
    return "Box(dim=" + std::to_string(dim()) + ")";
}

} // namespace rl::core
