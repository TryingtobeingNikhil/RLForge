# RLForge

> A from-scratch Reinforcement Learning library in modern C++20 — built incrementally, milestone by milestone, with production-quality engineering at every step.

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue.svg)](https://cmake.org/)
[![Catch2](https://img.shields.io/badge/Tests-Catch2%20v3-green.svg)](https://github.com/catchorg/Catch2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is RLForge?

RLForge is a reinforcement learning library built without shortcuts:

- **No black-box dependencies** — no PyTorch, no Eigen, no external tensor library. Every abstraction is written and understood from first principles.
- **Milestone-driven architecture** — each milestone introduces exactly what's needed, with documented design decisions and explicit deferral of complexity that isn't justified yet.
- **Production-quality engineering** — warnings-as-errors (`-Wall -Wextra -Wpedantic -Werror`), Catch2 test coverage for every component, and rich API documentation for every subsystem.
- **Designed for extension** — interfaces are built to accommodate future algorithms (DQN, PPO, SAC) and infrastructure (CUDA, multi-threaded rollout collection) without requiring rewrites of existing code.

---

## Architecture Overview

```
rl-lib/
├── include/rl/
│   ├── core/           # Environment, Space, Transition, ReplayBuffer, Agent, Trainer
│   ├── envs/           # GridWorld (reference environment)
│   ├── vector_envs/    # SyncVectorEnvironment
│   ├── replay_buffers/ # VectorTransitionStorage
│   ├── agents/         # TabularQLearningAgent
│   └── tensor/         # Tensor, autograd (Milestone 5)
├── src/                # Implementations (mirrors include/ structure)
├── tests/              # Catch2 test suite (single binary)
└── docs/               # Per-subsystem API documentation
```

---

## Milestones

### ✅ Milestone 1 — Environment Interface
*Foundation: what every RL environment must expose.*

- `Environment` abstract base class with `reset()` / `step()`.
- `Space` hierarchy: `Discrete(n)` and `Box(low, high)`.
- `StepResult` with **separate `terminated` / `truncated` flags** (Gymnasium-style) — critical for correct bootstrapping in value-based algorithms.
- `Observation`, `Action`, `Value` variant types.

📄 [`docs/environment_api.md`](docs/environment_api.md)

---

### ✅ Milestone 2 — Vector Environment
*Parallel environment interface for batched rollout collection.*

- `VectorEnvironment` abstract base — batched `reset()` / `step()` over N sub-environments.
- `SyncVectorEnvironment` — synchronous reference implementation.
- **Auto-reset semantics with `final_observations`**: terminal observations are preserved separately from the post-reset initial observation, preventing a silent but common bug in batched training loops.
- Contract tests shared across all `VectorEnvironment` implementations.

📄 [`docs/vector_environment_api.md`](docs/vector_environment_api.md)

---

### ✅ Milestone 3 — Replay Buffer & Transitions
*Experience replay for off-policy algorithms.*

- `Transition` struct with full `terminated` / `truncated` semantics.
- `ReplayBuffer` — fixed-capacity ring buffer with uniform random sampling.
- `TransitionStorage` interface: decouples buffer policy from physical layout, enabling future migration to flat contiguous float buffers without changing the sampling API.
- `VectorTransitionStorage` — reference backend.

📄 [`docs/replay_buffer_api.md`](docs/replay_buffer_api.md)

---

### ✅ Milestone 4 — Agent & Trainer Interface
*Universal training loop abstraction.*

- `Agent` abstract base — four deliberate hooks: `act()`, `observe_transitions()`, `should_update()`, `update()`. Decoupled so step-based algorithms (Q-Learning) and rollout-based algorithms (PPO) share the same `Trainer` without awkward adaptation.
- `Trainer` — drives the `VectorEnvironment → Agent` loop; separate `train()` and `evaluate()` paths.
- `TabularQLearningAgent` — reference implementation. Validates the full pipeline on `GridWorld`: learns the optimal 3-return policy in 20,000 steps, verified arithmetically not just empirically.

📄 [`docs/agent_trainer_api.md`](docs/agent_trainer_api.md)

---

### ✅ Milestone 5 — Tensor & Reverse-Mode Autograd
*The mathematical foundation for Deep RL.*

A self-contained tensor library and automatic differentiation engine — built without Eigen, xtensor, or PyTorch.

**Tensor features:**
- Dense, contiguous, row-major `double` storage via `shared_ptr<Storage>`.
- General N-D shape (`vector<int64_t>`); 1-D and 2-D fully supported this milestone.
- Ops: `add`, `sub`, `mul` (elementwise + scalar), `matmul` (2-D), `relu`, `square`, `mean`, `gather`.

**Autograd features:**
- Dynamic, define-by-run reverse-mode differentiation.
- Iterative (non-recursive) topological sort for the backward pass.
- **Correct diamond-dependency handling** — gradients from multiple downstream consumers accumulate into a shared buffer before the upstream node's `backward_fn` fires; not just "visit once" but "accumulate-all-consumers-first-then-call."
- `no_grad()` RAII guard — disables graph construction for inference/evaluation.
- `detach()`, `zero_grad()`, explicit gradient accumulation.
- `gather` backward is scatter-add with correct duplicate-index accumulation.
- Every `backward_fn` runs under `no_grad()` — gradient computations don't accidentally build a second graph on top of the backward pass.

**Test coverage:**
- Forward-value tests for every op.
- Analytical backward tests for every op.
- **Numerical gradient checking** (central difference, relative tolerance 1e-5) for every differentiable op.
- Diamond-dependency test.
- Gather-with-duplicate-indices test.
- End-to-end: `loss = mean((W @ x - target)²)` — loss value and `dL/dW` verified numerically.

📄 [`docs/tensor_autograd_api.md`](docs/tensor_autograd_api.md)

**Explicitly deferred (documented in header comments and docs):**
- General tensor-tensor broadcasting (needed for Milestone 6 `Linear` bias).
- Views / strided tensors.
- In-place mutation guard (⚠️ must add before any optimizer implementation).
- BLAS / SIMD / GPU acceleration.

---

## Roadmap

| Milestone | Status | Description |
|-----------|--------|-------------|
| 1 — Environment Interface | ✅ Done | `Environment`, `Space`, terminated/truncated semantics |
| 2 — Vector Environment | ✅ Done | `VectorEnvironment`, `SyncVectorEnvironment`, auto-reset |
| 3 — Replay Buffer | ✅ Done | `Transition`, `ReplayBuffer`, `TransitionStorage` |
| 4 — Agent & Trainer | ✅ Done | `Agent`, `Trainer`, `TabularQLearningAgent` |
| 5 — Tensor & Autograd | ✅ Done | Tensor, reverse-mode AD, numerical grad checks |
| 6 — Neural Network Layers | ✅ Done | `Linear`, `ReLU`, broadcasting, in-place mutation guard |
| 7 — DQN | ✅ Done | Deep Q-Network on `GridWorld` / Atari |
| 8 — PPO | 🔜 Planned | Proximal Policy Optimization |
| 9 — Multi-threaded Rollouts | 🔜 Planned | Parallel environment collection |
| 10 — CUDA / BLAS | 🔜 Planned | GPU tensor backend |

---

## Building

```bash
git clone https://github.com/TryingtobeingNikhil/RLForge.git
cd RLForge

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Run tests:**
```bash
cd build && ctest --output-on-failure
```

**Requirements:**
- C++20-capable compiler (Clang 14+ / GCC 12+)
- CMake 3.20+
- Internet connection on first build (Catch2 fetched via `FetchContent`)

---

## Test Suite

130 tests across all milestones — all pass.

```
100% tests passed, 0 tests failed out of 130
Total Test time (real) = ~0.5 sec
```

Tests are organized into one binary (`rl_tests`) and tagged by subsystem:
`[space]`, `[replay_buffer]`, `[tabular_q_learning]`, `[tensor]`, `[tensor][autograd]`, `[tensor][numgrad]`, `[tensor][integration]`

---

## Documentation

Each subsystem has a dedicated API doc in [`docs/`](docs/):

| Document | Covers |
|----------|--------|
| [`environment_api.md`](docs/environment_api.md) | Milestones 1–2: Environment, Space, VectorEnvironment |
| [`replay_buffer_api.md`](docs/replay_buffer_api.md) | Milestone 3: Transition, ReplayBuffer |
| [`agent_trainer_api.md`](docs/agent_trainer_api.md) | Milestone 4: Agent, Trainer, TabularQLearning |
| [`tensor_autograd_api.md`](docs/tensor_autograd_api.md) | Milestone 5: Tensor, autograd, limitations |

---

## Engineering Principles

- **Warnings are errors.** `-Wall -Wextra -Wpedantic -Werror` from day one.
- **Explicit over implicit.** `terminated` and `truncated` are separate flags. `auto_reset` preserves `final_observations`. Gradient accumulation is explicit — no auto-zero.
- **Interfaces designed for the algorithm, not the implementation.** `TransitionStorage` can swap backends without touching `ReplayBuffer`. `Agent` can swap algorithms without touching `Trainer`.
- **Deferred complexity is documented, not just skipped.** Every known limitation has a code comment explaining *what* must be fixed, *why*, and *which milestone* it belongs to.
