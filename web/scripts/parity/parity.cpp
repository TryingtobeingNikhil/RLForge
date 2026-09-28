// Native reference run for the website's GridWorld port (web/lib/gridworld.ts).
//
// Builds the real RLForge pipeline with the exact config from
// tests/test_trainer_grid_world.cpp:52-78 and prints checkpoints that
// web/tests/gridworld.test.ts compares against. Not part of the library or
// its test suite; build instructions are in README.md next to this file.
#include <cstdio>

#include "rl/agents/tabular_q_learning_agent.hpp"
#include "rl/core/trainer.hpp"
#include "rl/envs/grid_world.hpp"
#include "rl/vector_envs/sync_vector_environment.hpp"

using namespace rl;

int main() {
    envs::GridWorld::Config config;
    config.size = 5;
    config.max_episode_steps = 100;
    config.slip_probability = 0.0f;

    std::vector<vector_envs::EnvFactory> factories;
    for (int i = 0; i < 4; ++i) {
        factories.push_back([config]() -> std::unique_ptr<core::Environment> {
            return std::make_unique<envs::GridWorld>(config);
        });
    }
    vector_envs::SyncVectorEnvironment train_env(std::move(factories));
    envs::GridWorld eval_env(config);

    agents::TabularQLearningConfig agent_config;
    agent_config.learning_rate = 0.1f;
    agent_config.discount_factor = 0.99f;
    agent_config.epsilon_start = 1.0f;
    agent_config.epsilon_end = 0.05f;
    agent_config.epsilon_decay_steps = 20000;
    agent_config.seed = 0;
    agents::TabularQLearningAgent agent(train_env.action_space(), agent_config);

    core::Trainer trainer(train_env, eval_env, agent);

    auto baseline = trainer.evaluate(10, 1000);
    std::printf("step 0 eval %.1f\n", baseline.episode_returns.front());

    // Trainer::train only resets the vector env on its first call, so
    // training in 250-step chunks is identical to one train(20000, 42) call.
    size_t episodes = 0;
    for (int chunk = 1; chunk <= 80; ++chunk) {
        auto result = trainer.train(250, 42);
        episodes += result.episode_returns.size();
        auto eval = trainer.evaluate(10, 1000);
        std::printf("step %d eval %.1f episodes %zu\n", chunk * 250,
                    eval.episode_returns.front(), episodes);
    }

    std::printf("Q(0,0,a):");
    for (int64_t a = 0; a < 4; ++a) {
        std::printf(" %.9g", agent.q_value(core::Observation{std::vector<float>{0.0f, 0.0f}}, a));
    }
    std::printf("\n");
    return 0;
}
