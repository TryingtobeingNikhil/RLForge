#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>

#include "rl/agents/tabular_q_learning_agent.hpp"
#include "rl/core/trainer.hpp"
#include "rl/envs/grid_world.hpp"
#include "rl/vector_envs/sync_vector_environment.hpp"

using Catch::Approx;
using rl::agents::TabularQLearningAgent;
using rl::agents::TabularQLearningConfig;
using rl::core::Trainer;
using rl::envs::GridWorld;
using rl::vector_envs::EnvFactory;
using rl::vector_envs::SyncVectorEnvironment;

namespace {

float mean(const std::vector<float>& values) {
    if (values.empty()) {
        return 0.0f;
    }
    return std::accumulate(values.begin(), values.end(), 0.0f) /
           static_cast<float>(values.size());
}

std::vector<EnvFactory> make_grid_world_factories(size_t count, GridWorld::Config config) {
    std::vector<EnvFactory> factories;
    factories.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        factories.push_back([config]() -> std::unique_ptr<rl::core::Environment> {
            return std::make_unique<GridWorld>(config);
        });
    }
    return factories;
}

} // namespace

// This is the milestone's central claim, verified directly: the
// Environment -> VectorEnvironment -> Agent -> Trainer pipeline, built up
// over four milestones with no algorithm-specific shortcuts, is enough on
// its own to learn a working policy -- with a plain tabular agent that
// knows nothing about GridWorld specifically.
//
// Every seed here is fixed, so this test is fully deterministic: the exact
// numbers below (baseline == -100, post-training == 3.0) were verified
// empirically before being written down, not guessed at.
TEST_CASE("Trainer + TabularQLearningAgent learns GridWorld end-to-end", "[trainer][integration]") {
    GridWorld::Config config;
    config.size = 5;
    config.max_episode_steps = 100;
    config.slip_probability = 0.0f; // deterministic dynamics for a crisp, non-flaky assertion

    SyncVectorEnvironment train_env(make_grid_world_factories(4, config));
    GridWorld eval_env(config);

    TabularQLearningConfig agent_config;
    agent_config.learning_rate = 0.1f;
    agent_config.discount_factor = 0.99f;
    agent_config.epsilon_start = 1.0f;
    agent_config.epsilon_end = 0.05f;
    agent_config.epsilon_decay_steps = 20000;
    agent_config.seed = 0;
    TabularQLearningAgent agent(train_env.action_space(), agent_config);

    Trainer trainer(train_env, eval_env, agent);

    // Before any training: the Q-table is empty, so the agent's greedy
    // policy (ties broken toward action 0, "Up") walks into a wall and
    // sits there until the time limit -- 100 steps of -1 reward, every
    // episode, deterministically.
    auto baseline = trainer.evaluate(/*num_episodes=*/10, /*seed=*/1000);
    REQUIRE(mean(baseline.episode_returns) == Approx(-100.0f));

    auto train_result = trainer.train(/*num_steps=*/20000, /*seed=*/42);

    // One update() call per training step -- tabular Q-Learning's
    // should_update() is true whenever there's a pending transition, i.e.
    // every step.
    REQUIRE(train_result.update_metrics.size() == 20000);
    // Sanity check on episode accounting: many episodes should have
    // completed (early ones are slow/near-random, later ones near-optimal).
    REQUIRE(train_result.episode_returns.size() > 1000);
    REQUIRE(train_result.episode_returns.size() == train_result.episode_lengths.size());

    auto after_training = trainer.evaluate(/*num_episodes=*/10, /*seed=*/1000);

    // The true optimal return on a 5x5 GridWorld with no slip: the shortest
    // path from (0,0) to (4,4) is 8 moves (Manhattan distance), 7 of which
    // score -1 and the 8th (reaching the goal) scores +10 instead of -1:
    // 7*(-1) + 10 = 3. A fully converged greedy policy hits this exactly,
    // every evaluation episode, since dynamics are deterministic here.
    REQUIRE(mean(after_training.episode_returns) == Approx(3.0f));

    // The claim in plain terms: training took the agent from "never
    // reaches the goal" to "reaches the goal optimally every time".
    REQUIRE(mean(after_training.episode_returns) - mean(baseline.episode_returns) >
            50.0f);
}

// A lightweight smoke test that Trainer's use of VectorEnvironment isn't
// secretly assuming num_envs() > 1 -- the single-env case must work without
// any special-casing, since Trainer has no branch that checks num_envs().
TEST_CASE("Trainer works with a single-environment VectorEnvironment", "[trainer]") {
    GridWorld::Config config;
    config.size = 3;
    config.max_episode_steps = 50;

    SyncVectorEnvironment train_env(make_grid_world_factories(1, config));
    GridWorld eval_env(config);

    TabularQLearningAgent agent(train_env.action_space());
    Trainer trainer(train_env, eval_env, agent);

    auto train_result = trainer.train(/*num_steps=*/200, /*seed=*/7);
    REQUIRE(train_result.update_metrics.size() == 200);

    auto eval_result = trainer.evaluate(/*num_episodes=*/3, /*seed=*/0);
    REQUIRE(eval_result.episode_returns.size() == 3);
    REQUIRE(eval_result.episode_lengths.size() == 3);
}
