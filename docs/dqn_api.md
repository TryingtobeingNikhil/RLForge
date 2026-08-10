# Deep Q-Network (DQN)

This document describes the implementation of the Deep Q-Network (DQN) agent (Milestone 7).

## Overview

The DQN implementation closely follows the original algorithm proposed by Mnih et al. (2015). It provides a full deep reinforcement learning agent capable of learning control policies directly from continuous observations.

The DQN implementation is located in the `rl::agents` namespace and integrates seamlessly with the existing `rl::core::Agent` and `rl::core::Trainer` infrastructure.

## Key Components

### 1. DQNAgent (`include/rl/agents/dqn.hpp`)

The main entry point for the DQN algorithm. `DQNAgent` derives from `rl::core::Agent`.

-   **Online Network (`online_net_`)**: The neural network (a `QNetwork`) whose weights are updated at every training step via backpropagation.
-   **Target Network (`target_net_`)**: A frozen copy of the online network used to compute the TD targets. It prevents the moving target problem. Its parameters have `requires_grad=false`. It is hard-synced from the online network every `target_update_freq` steps.
-   **Optimizer**: The `Adam` optimizer is used to update the online network parameters.
-   **Replay Buffer**: Stores transitions for experience replay, breaking temporal correlations in the data. We use the algorithm-agnostic `VectorTransitionStorage` inside the agent.

#### Target Sync Version Safety
The `DQNAgent` safely syncs target parameters by copying data directly into the target tensors using `data_mutable()`. This explicitly bumps the target network's storage versions without affecting the version counters of the online network parameters, avoiding false positives from the autograd version guard.

### 2. QNetwork (`include/rl/nn/qnetwork.hpp`)

A specialized neural network architecture for action-value prediction.

-   Dynamically constructs a sequence of fully-connected (`Linear`) layers with `ReLU` activations.
-   The final layer has no activation function, allowing it to predict unbounded Q-values (which can be negative).
-   Implements `parameters()` to aggregate all `Linear` weights and biases into a single list for the optimizer.
-   Takes batched state inputs `[B, state_dim]` and produces batched Q-values `[B, num_actions]`.

### 3. EpsilonGreedyPolicy (`include/rl/agents/epsilon_greedy.hpp`)

Handles exploration vs. exploitation during action selection.

-   Implements a linear decay schedule from `eps_start` to `eps_end` over `eps_decay_steps`.
-   Takes a caller-supplied `std::mt19937` random number generator to ensure training is fully deterministic when seeded.
-   Argmax tie-breaking resolves to the lowest index (first occurrence) to maintain consistency with `max_last_dim()`.

### 4. Tensor Extensions

To support the DQN loss function, the `Tensor` autograd engine was extended with:
-   **`max_last_dim()`**: Reduces a `[B, C]` tensor to a `[B]` tensor by finding the maximum value in each row. The backward pass routes gradients through a one-hot mask corresponding to the argmax index (using first-occurrence tie-breaking).

### 5. `batch_to_tensors` Utility (`include/rl/data/batch_to_tensors.hpp`)

A crucial algorithm-agnostic bridge between the C++ `TransitionBatch` (returned by `ReplayBuffer`) and the `Tensor` engine.

-   Dynamically allocates batched `Tensor` objects (with `requires_grad=false`) containing states, actions, rewards, next states, and the termination mask.
-   **Terminated vs Truncated**: Correctly distinguishes between true MDP termination and time-limit truncation. The bootstrap target is only zeroed if `terminated` is true; `truncated` transitions preserve the bootstrap estimate, avoiding the classic "done" flag bug.

## Training Loop

The `DQNAgent::update()` method performs one step of DQN training:

1.  **Sample**: Draws a mini-batch of transitions from the replay buffer.
2.  **Convert**: Converts the batch to Tensors via `batch_to_tensors`.
3.  **Forward Pass (Online)**: Computes $Q(s_t, a_t)$ using `online_net_.forward()` and `.gather(actions)`.
4.  **Forward Pass (Target)**: Computes $\max_{a} Q_{target}(s_{t+1}, a)$ under `rl::tensor::no_grad()`.
5.  **Compute Targets**: Calculates the Bellman targets $y_i = r_i + \gamma \max_{a} Q_{target}(s_{t+1}, a) (1 - \text{terminated}_i)$. The targets are explicitly `.detach()`ed.
6.  **Loss**: Computes the Mean Squared Error (MSE) between online predictions and targets.
7.  **Optimize**: Zeros gradients, calls `loss.backward()`, and steps the `Adam` optimizer.
8.  **Sync Target**: If the step counter reaches `target_update_freq`, copies parameters from the online network to the target network.

## Testing

The implementation is validated by a rigorous test suite (`test_dqn.cpp`):
-   **Correctness**: `QNetwork` shape checks, `batch_to_tensors` termination masking verification, epsilon decay schedules.
-   **Target Network Stability**: Verifies that the agent can train for hundreds of steps without throwing exceptions, proving the interaction between multiple optimiser steps and target network synchronisations is safe.
-   **End-to-End**: Demonstrates convergence on a deterministic bandit environment, showing that the loss trends downward and the agent learns the correct greedy actions.
