#include <catch2/catch_test_macros.hpp>

#include <random>

#include "rl/core/space.hpp"

using rl::core::Box;
using rl::core::Discrete;
using rl::core::Value;

TEST_CASE("Discrete rejects a non-positive n", "[space]") {
    REQUIRE_THROWS_AS(Discrete(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Discrete(-1), std::invalid_argument);
}

TEST_CASE("Discrete::sample always produces an in-range value", "[space]") {
    Discrete space(4);
    std::mt19937 rng(0);
    for (int i = 0; i < 1000; ++i) {
        Value sampled = space.sample(rng);
        REQUIRE(space.contains(sampled));
        REQUIRE(std::get<int64_t>(sampled) >= 0);
        REQUIRE(std::get<int64_t>(sampled) < 4);
    }
}

TEST_CASE("Discrete::contains rejects out-of-range indices and wrong variant type", "[space]") {
    Discrete space(4);
    REQUIRE(space.contains(Value{int64_t{0}}));
    REQUIRE(space.contains(Value{int64_t{3}}));
    REQUIRE_FALSE(space.contains(Value{int64_t{4}}));
    REQUIRE_FALSE(space.contains(Value{int64_t{-1}}));
    REQUIRE_FALSE(space.contains(Value{std::vector<float>{0.0f}}));
}

TEST_CASE("Box rejects mismatched or inverted bounds", "[space]") {
    REQUIRE_THROWS_AS(Box({0.0f, 0.0f}, {1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(Box({1.0f}, {0.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(Box({}, {}), std::invalid_argument);
}

TEST_CASE("Box::sample always produces a value within bounds", "[space]") {
    Box space({-1.0f, 0.0f}, {1.0f, 2.0f});
    std::mt19937 rng(0);
    for (int i = 0; i < 1000; ++i) {
        Value sampled = space.sample(rng);
        REQUIRE(space.contains(sampled));
        const auto& vec = std::get<std::vector<float>>(sampled);
        REQUIRE(vec[0] >= -1.0f);
        REQUIRE(vec[0] <= 1.0f);
        REQUIRE(vec[1] >= 0.0f);
        REQUIRE(vec[1] <= 2.0f);
    }
}

TEST_CASE("Box::contains rejects wrong size and wrong variant type", "[space]") {
    Box space({0.0f, 0.0f}, {1.0f, 1.0f});
    REQUIRE(space.contains(Value{std::vector<float>{0.5f, 0.5f}}));
    REQUIRE_FALSE(space.contains(Value{std::vector<float>{0.5f}}));
    REQUIRE_FALSE(space.contains(Value{std::vector<float>{2.0f, 0.5f}}));
    REQUIRE_FALSE(space.contains(Value{int64_t{0}}));
}
