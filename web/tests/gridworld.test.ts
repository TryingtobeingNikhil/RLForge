import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { GRIDWORLD_CONFIG, GridWorldTrainer } from "@/lib/gridworld";

// The port must reproduce the two numbers asserted in
// tests/test_trainer_grid_world.cpp:76 (-100 before training) and :96 (3.0 after).
describe("GridWorld + tabular Q-learning port", () => {
  it("scores exactly -100 before training (greedy ties -> Up -> wall for 100 steps)", () => {
    const trainer = new GridWorldTrainer();
    expect(trainer.evaluate()).toBe(-100);
    const rollout = trainer.greedyRollout();
    expect(rollout.path).toHaveLength(GRIDWORLD_CONFIG.maxEpisodeSteps + 1);
    expect(rollout.reachedGoal).toBe(false);
    expect(rollout.path.at(-1)).toEqual({ x: 0, y: 4 });
  });

  it("converges to the optimal greedy return of 3.0 after 20,000 trainer steps", () => {
    const trainer = new GridWorldTrainer();
    while (!trainer.done) trainer.step();
    expect(trainer.trainStep).toBe(20000);
    expect(trainer.agent.stepCount).toBe(80000); // 20,000 steps x 4 lanes
    expect(trainer.agent.epsilon()).toBeCloseTo(0.05, 6);
    expect(trainer.episodesCompleted).toBeGreaterThan(1000); // test_trainer_grid_world.cpp:86
    expect(trainer.evaluate()).toBe(3);
    const rollout = trainer.greedyRollout();
    expect(rollout.reachedGoal).toBe(true);
    expect(rollout.path).toHaveLength(9); // start + 8 moves: 7 x (-1) + 10 = 3
  });

  it("matches the native C++ run checkpoint for checkpoint (scripts/parity/expected-libcxx.txt)", () => {
    const expected = fs
      .readFileSync(path.join(__dirname, "../scripts/parity/expected-libcxx.txt"), "utf8")
      .trim()
      .split("\n");
    const actual: string[] = [];
    const trainer = new GridWorldTrainer();
    actual.push(`step 0 eval ${trainer.evaluate().toFixed(1)}`);
    while (!trainer.done) {
      trainer.step();
      if (trainer.trainStep % 250 === 0) {
        actual.push(
          `step ${trainer.trainStep} eval ${trainer.evaluate().toFixed(1)} episodes ${trainer.episodesCompleted}`,
        );
      }
    }
    const q = [0, 1, 2, 3].map((a) => trainer.agent.qValue({ x: 0, y: 0 }, a).toPrecision(9));
    actual.push(`Q(0,0,a): ${q.join(" ")}`);
    expect(actual).toEqual(expected);
  });
});
