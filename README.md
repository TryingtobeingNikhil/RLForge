# RLForge

**We built the tensor engine so we didn't have to trust anyone else's gradients.**

[![tests](https://img.shields.io/badge/tests-148%2F148-brightgreen.svg)](#the-test-suite) [![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

---

## Read this before you judge the checkmarks

There's a version of this project that took a weekend: `pip install torch gymnasium`,
copy a DQN tutorial, change some hyperparameters, call it "reinforcement learning from
scratch." That library is not this library.

This one has no `import torch`. No `import numpy`. No Eigen, no xtensor, no borrowed
autograd. When you call `.backward()` in RLForge, the gradient that comes out is one
we derived, wrote, and numerically verified ourselves — because we wanted to know,
concretely, in code we could point to, *why* backpropagation is correct instead of
just trusting that it is.

That's a slower way to build an RL library. It is not the easier way. It is, we think,
the only way to actually understand one.

---

## The bug that's hiding in most "from scratch" autograd engines

Draw this graph:

```
        x
       / \
      a   b        a = x * 2,  b = x * 3
       \ /
        c          c = a + b
        |
       loss = mean(c)
```

`x` feeds two branches that reconverge at `c`. This is the single most common way a
homemade autograd engine silently produces the *wrong* gradient — not a crash, not an
exception, just a quietly incorrect number that trains a model into a local optimum
nobody asked for.

The failure mode is almost always the same: whoever writes the backward pass processes
`a`'s branch, computes `x`'s gradient, and moves on — overwriting instead of
accumulating when `b`'s branch reaches `x` a moment later. It works on every simple
example. It breaks the instant your graph isn't a straight line.

RLForge's backward pass is built around one non-negotiable rule: **a node is not
processed until every consumer that depends on it has already deposited its gradient
contribution.** That's what the topological sort and the gradient-accumulation buffer
in `tensor.cpp` actually exist to guarantee. There's a test for exactly this graph
shape. It's not decorative — it's the thing that tells us the rest of the math is
trustworthy.

---

## Two rules we refused to break

**1. `terminated` and `truncated` are never the same flag.**
An episode that *ends* because the agent died and an episode that gets *cut off*
because you hit a time limit look identical if you only track `done`. They are not
identical to the Bellman equation — one should zero out the bootstrap value, the other
should keep it. Collapse them into one boolean and your agent learns a value function
for a world that doesn't exist. Every environment, buffer, and batch conversion in this
library keeps them separate, on purpose, everywhere.

**2. A tensor's gradient is invalid the moment you mutate it in place after using it
in a forward pass — so we made that impossible to do silently.**
Every tensor's underlying storage carries a version counter. Every backward closure
captures the version it saw at forward time. If those don't match when `.backward()`
runs, RLForge throws instead of handing you a gradient computed against data that no
longer exists. Most from-scratch autograd projects skip this because it's annoying to
build. We built it because skipping it means your bugs show up as "training is
unstable" three weeks later instead of a stack trace today.

---

## The build log

| # | Milestone | The actual hard part |
|---|---|---|
| 1 | Environment Interface | Getting `terminated`/`truncated` right before anything else depended on getting it wrong |
| 2 | Vector Environment | Auto-reset that preserves the *real* final observation instead of quietly discarding it |
| 3 | Replay Buffer & Transitions | A storage interface the sampling logic doesn't need to know or care about |
| 4 | Agent & Trainer | One training loop that runs step-based Q-Learning and rollout-based PPO without either one bending to fit the other |
| 5 | Tensor & Autograd | The diamond-dependency problem above, solved and numerically proven, not assumed |
| 6 | NN Layers & Optimizers | Kaiming init, SGD with momentum, Adam, broadcasting — all running on our own tensor engine |
| 7 | DQN | Bootstrap masking that respects terminated vs. truncated all the way through the target computation |
| 8 | PPO | Generalized advantage estimation with lane-correct handling across a batch of parallel environments |
| 9 | Multi-threaded Rollouts | Persistent worker threads, deterministic ordering, and gradient-mode isolation that doesn't leak across threads |
| 10 | CUDA / CBLAS Backends | Swapping the matmul kernel underneath the whole autograd graph without the graph noticing |

Milestones 1–7 — the environment stack, the tensor and autograd engine, and DQN — were
built end-to-end by **Nikhil Mourya**. Milestones 8–10 — PPO, the threading layer, and
the CUDA/BLAS backends — were built together with **[aprv10](https://github.com/aprv10)**,
who led the concurrency and GPU work. Full breakdown in [Credits](#credits).

---

## Scars

We're not going to pretend this shipped clean the whole way.

`-Wall -Wextra -Wpedantic -Werror` has been on since commit one, and it earns its keep:
at one point a stray backslash at the end of a comment in a test file — meant as
harmless ASCII art — got read by the compiler as a line continuation and broke the
build on a completely fresh clone. `-Werror` caught it immediately. It's a small thing,
but it's the whole argument for treating warnings as errors: the bug that costs you
five minutes today is the one that would've cost someone else an afternoon of
"why won't this compile" next month.

We'd rather tell you that than pretend every commit was clean.

---

## Building it

```bash
git clone https://github.com/TryingtobeingNikhil/RLForge.git
cd RLForge

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Run the tests:**

```bash
cd build && ctest --output-on-failure
```

**You'll need:**

- A C++20 compiler (Clang 14+ / GCC 12+)
- CMake 3.20+
- An internet connection the first time — Catch2 is fetched via `FetchContent`

---

## The test suite

148 tests, one binary, tagged by subsystem. Every tensor op has a forward test, an
analytical backward test, *and* a numerical gradient check (central difference,
1e-5 tolerance) — because "the math looks right" and "the math checks out numerically"
are different claims, and only one of them is a test. Every RL algorithm has an
end-to-end test that verifies actual convergence on a solvable environment, not just
that the code runs without throwing.

Don't take our word for any of this. Clone it, build it, run `ctest` yourself.

---

## Documentation

| Doc | Covers |
|---|---|
| [`environment_api.md`](./docs/environment_api.md) | Environment, Space, VectorEnvironment |
| [`replay_buffer_api.md`](./docs/replay_buffer_api.md) | Transition, ReplayBuffer |
| [`agent_trainer_api.md`](./docs/agent_trainer_api.md) | Agent, Trainer, TabularQLearning |
| [`tensor_autograd_api.md`](./docs/tensor_autograd_api.md) | Tensor, autograd, and its documented limitations |
| [`nn_optim_api.md`](./docs/nn_optim_api.md) | Neural layers and optimizers |
| [`dqn_api.md`](./docs/dqn_api.md) | Deep Q-Network |
| [`ppo_api.md`](./docs/ppo_api.md) | PPO and generalized advantage estimation |
| [`threaded_vector_environment_api.md`](./docs/threaded_vector_environment_api.md) | Persistent threaded rollouts |
| [`tensor_backends_api.md`](./docs/tensor_backends_api.md) | CPU, CBLAS, and CUDA backends |

---

## What we didn't build (yet)

Documented, not hidden — every deferred piece has a comment explaining what it is,
why it's not here, and which milestone it belongs to when it lands:

- Strided / view tensors
- General N-D broadcasting beyond what DQN and PPO need today
- Continuous-action policies (everything here is discrete-action for now)

If a homemade library's README doesn't have a section like this, it either doesn't
have limitations or isn't telling you about them. We know which one is true here.

---

## Credits

Designed and primarily built by **Nikhil Mourya**
([@TryingtobeingNikhil](https://github.com/TryingtobeingNikhil)) — architecture, the
environment/replay buffer/trainer stack, the tensor and autograd engine, NN layers and
optimizers, and DQN.

PPO, the multi-threaded rollout system, and the CUDA/BLAS backend were built together
with **[aprv10](https://github.com/aprv10)**.

## License

MIT — see [`LICENSE`](./LICENSE).
