#include <catch2/catch_test_macros.hpp>

#include <random>
#include <set>

#include "rl/core/replay_buffer.hpp"
#include "rl/replay_buffers/vector_transition_storage.hpp"

using rl::core::ReplayBuffer;
using rl::core::Transition;
using rl::core::TransitionStorage;
using rl::replay_buffers::VectorTransitionStorage;

namespace {

// A transition distinguishable only by its reward, for tests that just need
// to track "which transitions ended up where" without caring about
// observation/action content.
Transition make_marker_transition(float reward) {
    Transition transition;
    transition.reward = reward;
    return transition;
}

// A second, independent TransitionStorage implementation used purely to
// prove that ReplayBuffer only ever interacts with storage through the
// TransitionStorage interface -- not just with the bundled
// VectorTransitionStorage. If ReplayBuffer worked with this one too, the
// "swap the backend without changing the public API" property actually
// holds, rather than being merely asserted in documentation.
class RecordingTransitionStorage final : public TransitionStorage {
public:
    explicit RecordingTransitionStorage(size_t capacity) : storage_(capacity) {}

    void set(size_t index, Transition transition) override {
        ++set_calls;
        storage_.at(index) = std::move(transition);
    }
    Transition get(size_t index) const override {
        ++get_calls;
        return storage_.at(index);
    }
    size_t capacity() const override { return storage_.size(); }

    mutable size_t get_calls = 0;
    size_t set_calls = 0;

private:
    std::vector<Transition> storage_;
};

} // namespace

TEST_CASE("ReplayBuffer derives capacity from its storage backend", "[replay_buffer]") {
    ReplayBuffer buffer(std::make_unique<VectorTransitionStorage>(10));
    REQUIRE(buffer.capacity() == 10);
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.empty());
}

TEST_CASE("ReplayBuffer rejects a null storage backend", "[replay_buffer]") {
    std::unique_ptr<TransitionStorage> null_storage;
    REQUIRE_THROWS_AS(ReplayBuffer(std::move(null_storage)), std::invalid_argument);
}

TEST_CASE("VectorTransitionStorage rejects zero capacity", "[replay_buffer]") {
    REQUIRE_THROWS_AS(VectorTransitionStorage(0), std::invalid_argument);
}

TEST_CASE("ReplayBuffer::add tracks size up to capacity, then overwrites oldest (ring buffer)", "[replay_buffer]") {
    auto storage = std::make_unique<VectorTransitionStorage>(3);
    // Keep a raw observing pointer so the test can inspect exact slot
    // contents after ownership moves into the ReplayBuffer -- possible only
    // because get() is part of the public TransitionStorage interface.
    VectorTransitionStorage* storage_view = storage.get();

    ReplayBuffer buffer(std::move(storage));
    REQUIRE(buffer.capacity() == 3);

    buffer.add(make_marker_transition(0.0f));
    buffer.add(make_marker_transition(1.0f));
    REQUIRE(buffer.size() == 2);

    buffer.add(make_marker_transition(2.0f));
    REQUIRE(buffer.size() == 3);

    // Buffer is now full; the next two adds must wrap around and overwrite
    // slots 0 and 1, leaving slot 2 (reward 2.0) untouched.
    buffer.add(make_marker_transition(3.0f));
    buffer.add(make_marker_transition(4.0f));

    REQUIRE(buffer.size() == 3); // capped at capacity, never grows past it
    REQUIRE(storage_view->get(0).reward == 3.0f);
    REQUIRE(storage_view->get(1).reward == 4.0f);
    REQUIRE(storage_view->get(2).reward == 2.0f);
}

TEST_CASE("ReplayBuffer::sample on an empty buffer throws", "[replay_buffer]") {
    ReplayBuffer buffer(std::make_unique<VectorTransitionStorage>(5));
    std::mt19937 rng(0);
    REQUIRE_THROWS_AS(buffer.sample(4, rng), std::logic_error);
}

TEST_CASE("ReplayBuffer::sample returns correctly-sized batches drawn from stored transitions", "[replay_buffer]") {
    ReplayBuffer buffer(std::make_unique<VectorTransitionStorage>(5));
    for (float reward = 0.0f; reward < 5.0f; reward += 1.0f) {
        buffer.add(make_marker_transition(reward));
    }

    std::mt19937 rng(42);
    auto batch = buffer.sample(200, rng);

    REQUIRE(batch.observations.size() == 200);
    REQUIRE(batch.actions.size() == 200);
    REQUIRE(batch.rewards.size() == 200);
    REQUIRE(batch.next_observations.size() == 200);
    REQUIRE(batch.terminated.size() == 200);
    REQUIRE(batch.truncated.size() == 200);

    std::set<float> distinct_rewards_seen(batch.rewards.begin(), batch.rewards.end());
    for (float reward : distinct_rewards_seen) {
        REQUIRE(reward >= 0.0f);
        REQUIRE(reward <= 4.0f);
    }
    // With 200 draws (with replacement) from 5 items, seeing only one
    // distinct value would be an astronomically unlikely coincidence --
    // this is a sanity check that sampling is not just always returning
    // the same index, not a rigorous statistical test.
    REQUIRE(distinct_rewards_seen.size() > 1);
}

TEST_CASE("ReplayBuffer::sample with the same RNG state produces the same batch", "[replay_buffer]") {
    auto make_populated_buffer = []() {
        ReplayBuffer buffer(std::make_unique<VectorTransitionStorage>(5));
        for (float reward = 0.0f; reward < 5.0f; reward += 1.0f) {
            buffer.add(make_marker_transition(reward));
        }
        return buffer;
    };

    ReplayBuffer buffer_a = make_populated_buffer();
    ReplayBuffer buffer_b = make_populated_buffer();

    std::mt19937 rng_a(7);
    std::mt19937 rng_b(7);
    auto batch_a = buffer_a.sample(50, rng_a);
    auto batch_b = buffer_b.sample(50, rng_b);

    REQUIRE(batch_a.rewards == batch_b.rewards);
}

TEST_CASE("ReplayBuffer works against a TransitionStorage implementation other than the bundled one", "[replay_buffer]") {
    auto storage = std::make_unique<RecordingTransitionStorage>(4);
    RecordingTransitionStorage* storage_view = storage.get();

    ReplayBuffer buffer(std::move(storage));
    buffer.add(make_marker_transition(1.0f));
    buffer.add(make_marker_transition(2.0f));

    std::mt19937 rng(0);
    auto batch = buffer.sample(10, rng);

    REQUIRE(batch.rewards.size() == 10);
    REQUIRE(storage_view->set_calls == 2);
    REQUIRE(storage_view->get_calls == 10);
}
