#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "rl/agents/tabular_q_learning_agent.hpp"
#include "rl/core/space.hpp"
#include "rl/core/types.hpp"

using Catch::Approx;
using rl::agents::TabularQLearningAgent;
using rl::agents::TabularQLearningConfig;
using rl::core::Action;
using rl::core::Box;
using rl::core::Discrete;
using rl::core::Observation;
using rl::core::Transition;

TEST_CASE("TabularQLearningAgent requires a Discrete action space", "[tabular_q_learning]") {
    Box continuous_action_space({0.0f}, {1.0f});
    REQUIRE_THROWS_AS(TabularQLearningAgent(continuous_action_space), std::invalid_argument);
}

TEST_CASE("TabularQLearningAgent acts greedily with ties broken toward the lowest action index", "[tabular_q_learning]") {
    Discrete action_space(4);
    TabularQLearningAgent agent(action_space);

    // Fresh agent, empty Q-table -> every action reads as 0.0, so the
    // greedy choice (explore=false) must consistently be action 0.
    auto actions = agent.act(
        {Observation{int64_t{0}}, Observation{std::vector<float>{3.0f, 1.0f}}},
        /*explore=*/false);

    REQUIRE(actions.size() == 2);
    REQUIRE(rl::core::as_index(actions[0]) == 0);
    REQUIRE(rl::core::as_index(actions[1]) == 0);
}

TEST_CASE("TabularQLearningAgent does not bootstrap across a terminated transition", "[tabular_q_learning]") {
    Discrete action_space(2);
    TabularQLearningConfig config;
    config.learning_rate = 0.5f;
    config.discount_factor = 0.9f;
    TabularQLearningAgent agent(action_space, config);

    const Observation state0{int64_t{0}};
    const Observation state1{int64_t{1}};

    // Seed Q(state1, action=0) to a large, easily-distinguished value, so
    // we can verify below that it is correctly IGNORED when a transition
    // into state1 is marked terminated.
    Transition seed;
    seed.observation = state1;
    seed.action = Action{int64_t{0}};
    seed.reward = 100.0f;
    seed.next_observation = state1; // irrelevant: terminated=true ignores it
    seed.terminated = true;
    agent.observe_transitions({seed});
    agent.update();
    REQUIRE(agent.q_value(state1, 0) == Approx(50.0f)); // 0 + 0.5*(100 - 0)

    // A transition INTO state1 that is itself terminated. Its TD target
    // must be just the reward -- no gamma * max_a Q(state1, a) term, even
    // though that value is now a large 50.0.
    Transition terminated_transition;
    terminated_transition.observation = state0;
    terminated_transition.action = Action{int64_t{0}};
    terminated_transition.reward = 5.0f;
    terminated_transition.next_observation = state1;
    terminated_transition.terminated = true;
    agent.observe_transitions({terminated_transition});
    agent.update();

    // target = reward + 0 (terminated) = 5.0; Q(state0,0) = 0 + 0.5*(5-0)
    REQUIRE(agent.q_value(state0, 0) == Approx(2.5f));
}

TEST_CASE("TabularQLearningAgent bootstraps across a truncated transition", "[tabular_q_learning]") {
    Discrete action_space(2);
    TabularQLearningConfig config;
    config.learning_rate = 0.5f;
    config.discount_factor = 0.9f;
    TabularQLearningAgent agent(action_space, config);

    const Observation state0{int64_t{0}};
    const Observation state1{int64_t{1}};

    Transition seed;
    seed.observation = state1;
    seed.action = Action{int64_t{0}};
    seed.reward = 100.0f;
    seed.next_observation = state1;
    seed.terminated = true;
    agent.observe_transitions({seed});
    agent.update();
    REQUIRE(agent.q_value(state1, 0) == Approx(50.0f));

    // Same setup as the terminated test above, EXCEPT this transition is
    // truncated, not terminated -- the underlying MDP did not actually end,
    // so its TD target must include the bootstrap term.
    Transition truncated_transition;
    truncated_transition.observation = state0;
    truncated_transition.action = Action{int64_t{0}};
    truncated_transition.reward = 5.0f;
    truncated_transition.next_observation = state1;
    truncated_transition.terminated = false;
    truncated_transition.truncated = true;
    agent.observe_transitions({truncated_transition});
    agent.update();

    // target = reward + gamma * max_a Q(state1,a) = 5.0 + 0.9*50.0 = 50.0
    // Q(state0,0) = 0 + 0.5*(50.0 - 0) = 25.0
    REQUIRE(agent.q_value(state0, 0) == Approx(25.0f));
}

TEST_CASE("TabularQLearningAgent's reported epsilon decays linearly and clamps at epsilon_end", "[tabular_q_learning]") {
    Discrete action_space(2);
    TabularQLearningConfig config;
    config.epsilon_start = 1.0f;
    config.epsilon_end = 0.0f;
    config.epsilon_decay_steps = 10;
    TabularQLearningAgent agent(action_space, config);

    Transition dummy;
    dummy.observation = Observation{int64_t{0}};
    dummy.action = Action{int64_t{0}};
    dummy.next_observation = Observation{int64_t{0}};
    dummy.terminated = true;

    for (int step = 1; step <= 10; ++step) {
        agent.observe_transitions({dummy});
        auto metrics = agent.update();
        const double epsilon = std::get<double>(metrics.at("epsilon"));
        const double expected = 1.0 - static_cast<double>(step) / 10.0;
        REQUIRE(epsilon == Approx(expected));
    }

    // Beyond epsilon_decay_steps, epsilon must clamp at epsilon_end rather
    // than continue past it (or go negative, if it were extrapolated).
    agent.observe_transitions({dummy});
    auto metrics = agent.update();
    REQUIRE(std::get<double>(metrics.at("epsilon")) == Approx(0.0));
}

TEST_CASE("TabularQLearningAgent::update reports q_table_size growing as new states are visited", "[tabular_q_learning]") {
    Discrete action_space(2);
    TabularQLearningAgent agent(action_space);
    REQUIRE(agent.table_size() == 0);

    Transition transition;
    transition.observation = Observation{int64_t{0}};
    transition.action = Action{int64_t{0}};
    transition.next_observation = Observation{int64_t{1}};
    transition.terminated = true;
    agent.observe_transitions({transition});
    agent.update();

    REQUIRE(agent.table_size() == 1);
}
