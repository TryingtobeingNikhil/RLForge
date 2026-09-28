# GridWorld parity harness

`parity.cpp` runs the real RLForge pipeline (GridWorld, SyncVectorEnvironment,
TabularQLearningAgent, Trainer) with the exact config from
`tests/test_trainer_grid_world.cpp` and prints checkpoints every 250 trainer
steps. `web/tests/gridworld.test.ts` asserts that the TypeScript port in
`web/lib/gridworld.ts` hits the same checkpoints.

Build and run from the repository root:

```bash
clang++ -std=c++20 -O2 -Iinclude web/scripts/parity/parity.cpp \
  src/core/*.cpp src/envs/grid_world.cpp \
  src/vector_envs/sync_vector_environment.cpp \
  src/agents/tabular_q_learning_agent.cpp -o /tmp/rlforge-parity
/tmp/rlforge-parity
```

The full output from macOS with Apple clang (libc++), whose distributions the
port follows, is checked in as `expected-libcxx.txt`. Excerpt:

```text
step 0 eval -100.0
step 250 eval -100.0 episodes 16
step 500 eval -100.0 episodes 38
step 750 eval 3.0 episodes 65
step 1000 eval 3.0 episodes 88
...
step 20000 eval 3.0 episodes 8408
Q(0,0,a): 2.52717257 1.50190043 1.50190043 2.52717257
```

A GCC/libstdc++ build uses different `uniform_*_distribution` algorithms, so
its exploration trajectory differs. It still starts at -100 and ends at 3.0,
which is all `tests/test_trainer_grid_world.cpp` asserts.
