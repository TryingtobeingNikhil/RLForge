# Neural Network & Optimizer API

Milestone 6 introduces the high-level neural network and optimisation layers in `rlForge` (the `rl::nn` and `rl::optim` namespaces). These sit on top of the Milestone 5 tensor autograd engine, providing the abstractions necessary for reinforcement learning agents to learn via gradient descent.

## `rl::nn::Module`

The `Module` class is the abstract base class for all neural network layers (and eventually, entire networks).

### API Contract

Every module subclass must implement:
1. `forward(Tensor)`: Computes the layer's output from an input tensor.
2. `parameters()`: Returns all learnable parameter tensors as a `std::vector<std::shared_ptr<Tensor>>`. 

**Why `shared_ptr<Tensor>`?**
Optimisers need a stable reference to the underlying parameters so they can update them in-place across multiple training steps without having to re-fetch them every iteration.

## Available Layers

### `rl::nn::Linear`
A fully-connected (affine) layer.

*   **Formula**: $y = xW^T + b$
*   **Initialisation**: 
    *   Weights: He (Kaiming) Normal initialisation.
    *   Biases: Zero initialisation.
*   **Broadcasting**: The bias addition uses `Tensor::add`'s broadcasting rules to add a `[out_features]` bias to a `[batch_size, out_features]` activation matrix.

## Loss Functions

Loss functions are implemented as free functions in `include/rl/nn/losses.hpp`. They are pure composition of tensor operations, meaning the autograd engine handles their backward passes automatically.

### `rl::nn::mse_loss`
Mean Squared Error.
*   **Formula**: $\text{mean}((pred - target)^2)$
*   **Usage**: `auto loss = rl::nn::mse_loss(predictions, targets);`

## `rl::optim::Optimizer`

The `Optimizer` class is the abstract base class for all parameter update rules.

### API Contract

1.  **Construction**: Accepts the `std::vector<std::shared_ptr<Tensor>>` returned by a Module's `parameters()` method.
2.  **`step()`**: Updates all parameters based on their current `.grad()` buffers.
3.  **`zero_grad()`**: Clears the gradient buffers of all parameters. This must be called before the `backward()` pass of the next iteration to prevent gradients from accumulating continuously.

### Version Guard (Safety Feature)

When an optimiser calls `step()`, it modifies the parameter tensors in-place. The tensor engine uses a **version counter guard** to prevent corrupting computation graphs.
If you build a graph (via `forward()`), update parameters in-place (via `step()`), and then attempt to call `backward()` on the old graph, the engine will throw a `std::runtime_error` complaining about a stale version. This prevents silent gradient calculation errors.

## Available Optimisers

### `rl::optim::SGD`
Stochastic Gradient Descent, with optional momentum.

*   `lr`: Learning rate.
*   `momentum`: Momentum factor (default `0.0`).

### `rl::optim::Adam`
Adaptive Moment Estimation (Kingma & Ba, 2015).

*   Includes standard bias correction.
*   Hyperparameters default to the paper's recommendations (`lr=0.001`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`).

## Example Usage (End-to-End)

```cpp
#include "rl/nn/linear.hpp"
#include "rl/nn/losses.hpp"
#include "rl/optim/adam.hpp"

// 1. Initialise model and optimiser
rl::nn::Linear layer(input_dim, output_dim);
rl::optim::Adam optimizer(layer.parameters(), /*lr=*/0.01);

for (int step = 0; step < 1000; ++step) {
    // 2. Forward pass
    auto predictions = layer.forward(batch_inputs);
    auto loss = rl::nn::mse_loss(predictions, batch_targets);

    // 3. Clear old gradients
    optimizer.zero_grad();
    
    // 4. Backward pass (compute new gradients)
    loss.backward();

    // 5. Update parameters
    optimizer.step();
}
```
