// A TypeScript port of the exact pipeline behind RLForge's GridWorld result:
//   GridWorld (src/envs/grid_world.cpp)
//   -> SyncVectorEnvironment with auto-reset (src/vector_envs/sync_vector_environment.cpp)
//   -> make_transitions (src/core/transition.cpp)
//   -> TabularQLearningAgent (src/agents/tabular_q_learning_agent.cpp)
//   -> Trainer::train / Trainer::evaluate (src/core/trainer.cpp)
//
// This is NOT the C++ library running in the browser. It is the same
// algorithm and the same config, re-implemented so it can run on a page.
// C++ does this arithmetic in float32, so every value here is rounded with
// Math.fround at the same points.

import { MT19937, uniformFloat01, uniformIntBelow } from "./mt19937";

const f = Math.fround;

export const GRIDWORLD_CONFIG = {
  size: 5, // tests/test_trainer_grid_world.cpp:53
  maxEpisodeSteps: 100, // tests/test_trainer_grid_world.cpp:54
  slipProbability: 0, // tests/test_trainer_grid_world.cpp:55
  numEnvs: 4, // tests/test_trainer_grid_world.cpp:57 (make_grid_world_factories(4, config))
  learningRate: 0.1, // tests/test_trainer_grid_world.cpp:61
  discountFactor: 0.99, // tests/test_trainer_grid_world.cpp:62
  epsilonStart: 1.0, // tests/test_trainer_grid_world.cpp:63
  epsilonEnd: 0.05, // tests/test_trainer_grid_world.cpp:64
  epsilonDecaySteps: 20000, // tests/test_trainer_grid_world.cpp:65 (counted in observed transitions: src/agents/tabular_q_learning_agent.cpp:105)
  agentSeed: 0, // tests/test_trainer_grid_world.cpp:66
  trainSteps: 20000, // tests/test_trainer_grid_world.cpp:78
  trainSeed: 42, // tests/test_trainer_grid_world.cpp:78
  evalEpisodes: 10, // tests/test_trainer_grid_world.cpp:75
  evalSeed: 1000, // tests/test_trainer_grid_world.cpp:75
  stepReward: -1, // src/envs/grid_world.cpp:90
  goalReward: 10, // src/envs/grid_world.cpp:90
} as const;

export type Action = 0 | 1 | 2 | 3;
// include/rl/envs/grid_world.hpp:46 — enum class GridAction { Up = 0, Down = 1, Left = 2, Right = 3 }
export const ACTION_NAMES = ["Up", "Down", "Left", "Right"] as const;
export const NUM_ACTIONS = 4;

export interface Obs {
  x: number;
  y: number;
}

interface StepResult {
  obs: Obs;
  reward: number;
  terminated: boolean;
  truncated: boolean;
}

// src/envs/grid_world.cpp
export class GridWorld {
  x = 0;
  y = 0;
  private steps = 0;
  private readonly rng = new MT19937();

  constructor(
    readonly size: number = GRIDWORLD_CONFIG.size,
    readonly maxEpisodeSteps: number = GRIDWORLD_CONFIG.maxEpisodeSteps,
    readonly slipProbability: number = GRIDWORLD_CONFIG.slipProbability,
  ) {}

  reset(seed?: number): Obs {
    if (seed !== undefined) this.rng.seed(seed);
    this.x = 0;
    this.y = 0;
    this.steps = 0;
    return { x: 0, y: 0 };
  }

  step(action: number): StepResult {
    let requested = action;
    // The slip roll is drawn every step, even at slip 0 (grid_world.cpp:62-66).
    if (uniformFloat01(this.rng) < f(this.slipProbability)) {
      requested = uniformIntBelow(this.rng, NUM_ACTIONS);
    }
    const max = this.size - 1;
    const clamp = (v: number) => Math.min(Math.max(v, 0), max);
    switch (requested) {
      case 0: this.y = clamp(this.y + 1); break; // Up
      case 1: this.y = clamp(this.y - 1); break; // Down
      case 2: this.x = clamp(this.x - 1); break; // Left
      case 3: this.x = clamp(this.x + 1); break; // Right
    }
    this.steps++;
    const terminated = this.x === max && this.y === max;
    const truncated = !terminated && this.steps >= this.maxEpisodeSteps;
    const reward = terminated ? GRIDWORLD_CONFIG.goalReward : GRIDWORLD_CONFIG.stepReward;
    return { obs: { x: this.x, y: this.y }, reward, terminated, truncated };
  }
}

interface Transition {
  obs: Obs;
  action: number;
  reward: number;
  next: Obs;
  terminated: boolean;
}

// src/agents/tabular_q_learning_agent.cpp. The C++ table is an
// unordered_map keyed by (observation, action) that defaults to 0.0f for
// unvisited pairs; a dense zero-initialised Float32Array is equivalent here
// because GridWorld only ever emits the 25 integer cells.
export class TabularQLearningAgent {
  readonly q: Float32Array;
  stepCount = 0;
  private readonly rng: MT19937;

  constructor(readonly size: number = GRIDWORLD_CONFIG.size, seed: number = GRIDWORLD_CONFIG.agentSeed) {
    this.q = new Float32Array(size * size * NUM_ACTIONS);
    this.rng = new MT19937(seed);
  }

  private idx(o: Obs, a: number) {
    return (o.y * this.size + o.x) * NUM_ACTIONS + a;
  }

  qValue(o: Obs, a: number): number {
    return this.q[this.idx(o, a)];
  }

  // Ties go to the lowest action index: strict '>' (tabular_q_learning_agent.cpp:56).
  bestActionAndValue(o: Obs): [number, number] {
    let best = 0;
    let bestValue = this.qValue(o, 0);
    for (let a = 1; a < NUM_ACTIONS; a++) {
      const v = this.qValue(o, a);
      if (v > bestValue) {
        bestValue = v;
        best = a;
      }
    }
    return [best, bestValue];
  }

  // Linear decay, clamped at epsilon_end (tabular_q_learning_agent.cpp:74-81).
  epsilon(): number {
    const c = GRIDWORLD_CONFIG;
    if (this.stepCount >= c.epsilonDecaySteps) return f(c.epsilonEnd);
    const fraction = f(f(this.stepCount) / f(c.epsilonDecaySteps));
    return f(f(c.epsilonStart) + fraction * f(f(c.epsilonEnd) - f(c.epsilonStart)));
  }

  act(observations: Obs[], explore: boolean): number[] {
    return observations.map((o) => {
      if (explore && uniformFloat01(this.rng) < this.epsilon()) {
        return uniformIntBelow(this.rng, NUM_ACTIONS);
      }
      return this.bestActionAndValue(o)[0];
    });
  }

  // observe_transitions() + update(): the Trainer calls both every step
  // because should_update() is true whenever transitions are pending.
  learn(transitions: Transition[]): void {
    const c = GRIDWORLD_CONFIG;
    this.stepCount += transitions.length;
    for (const t of transitions) {
      const current = this.qValue(t.obs, t.action);
      // tabular_q_learning_agent.cpp:120-122: no bootstrap across a terminated transition.
      const nextValue = t.terminated ? 0 : this.bestActionAndValue(t.next)[1];
      const target = f(f(t.reward) + f(c.discountFactor) * nextValue);
      const tdError = f(target - current);
      this.q[this.idx(t.obs, t.action)] = f(current + f(c.learningRate) * tdError);
    }
  }
}

export interface EvalPoint {
  step: number;
  meanReturn: number;
}

// src/core/trainer.cpp, driving a SyncVectorEnvironment of GRIDWORLD_CONFIG.numEnvs lanes.
export class GridWorldTrainer {
  readonly envs: GridWorld[];
  readonly evalEnv = new GridWorld();
  readonly agent = new TabularQLearningAgent();
  observations: Obs[] = [];
  trainStep = 0;
  episodesCompleted = 0;
  private started = false;
  private returns: number[];

  constructor(readonly numEnvs: number = GRIDWORLD_CONFIG.numEnvs) {
    this.envs = Array.from({ length: numEnvs }, () => new GridWorld());
    this.returns = new Array(numEnvs).fill(0);
  }

  get done(): boolean {
    return this.trainStep >= GRIDWORLD_CONFIG.trainSteps;
  }

  /** One Trainer::train loop iteration: act, step every lane, learn. */
  step(): void {
    if (!this.started) {
      // VectorEnvironment::reset(seed) gives sub-env i the seed `seed + i`.
      this.observations = this.envs.map((env, i) => env.reset(GRIDWORLD_CONFIG.trainSeed + i));
      this.started = true;
    }
    const actions = this.agent.act(this.observations, true);
    const transitions: Transition[] = [];
    const nextObservations: Obs[] = [];
    this.envs.forEach((env, i) => {
      const r = env.step(actions[i]);
      const ended = r.terminated || r.truncated;
      // make_transitions: the transition keeps the TRUE final observation,
      // while the lane itself auto-resets (no seed) for the next action.
      transitions.push({
        obs: this.observations[i],
        action: actions[i],
        reward: r.reward,
        next: r.obs,
        terminated: r.terminated,
      });
      nextObservations.push(ended ? env.reset() : r.obs);
      this.returns[i] = f(this.returns[i] + r.reward);
      if (ended) {
        this.episodesCompleted++;
        this.returns[i] = 0;
      }
    });
    this.agent.learn(transitions);
    this.observations = nextObservations;
    this.trainStep++;
  }

  /** Trainer::evaluate(10, 1000): greedy policy on a separate, non-vectorized env. */
  evaluate(
    numEpisodes: number = GRIDWORLD_CONFIG.evalEpisodes,
    seed: number = GRIDWORLD_CONFIG.evalSeed,
  ): number {
    let sum = 0;
    for (let ep = 0; ep < numEpisodes; ep++) {
      sum = f(sum + this.greedyRollout(seed + ep).return);
    }
    return f(sum / numEpisodes);
  }

  /** One greedy episode from (0,0); also returns the visited cells for drawing. */
  greedyRollout(seed: number = GRIDWORLD_CONFIG.evalSeed): { return: number; path: Obs[]; reachedGoal: boolean } {
    let obs = this.evalEnv.reset(seed);
    const path: Obs[] = [obs];
    let ret = 0;
    for (;;) {
      const [action] = this.agent.act([obs], false);
      const r = this.evalEnv.step(action);
      ret = f(ret + r.reward);
      obs = r.obs;
      path.push(obs);
      if (r.terminated || r.truncated) return { return: ret, path, reachedGoal: r.terminated };
    }
  }
}
