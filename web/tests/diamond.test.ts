import { describe, expect, it } from "vitest";
import { backward, forward, numericalGradient, relativeError } from "@/lib/diamond";

describe("diamond graph", () => {
  it("forward pass: c = 5x, loss = mean(c) = 12.5", () => {
    const f = forward();
    expect(f.c).toEqual([5, 10, 15, 20]);
    expect(f.loss).toBe(12.5);
  });

  it("accumulate: x.grad is 5/4 = 1.25 per element (tests/test_tensor.cpp:635)", () => {
    expect(backward("accumulate").grads.x).toEqual([1.25, 1.25, 1.25, 1.25]);
  });

  it("overwrite: b's deposit replaces a's and x.grad is 0.75, silently wrong", () => {
    const trace = backward("overwrite");
    expect(trace.grads.x).toEqual([0.75, 0.75, 0.75, 0.75]);
    const intoX = trace.deposits.filter((d) => d.to === "x");
    expect(intoX.map((d) => d.contribution[0])).toEqual([0.5, 0.75]);
  });

  it("the numerical gradient agrees with accumulate within 1e-5 and rejects overwrite", () => {
    const num = numericalGradient();
    for (const n of num) {
      expect(relativeError(1.25, n)).toBeLessThan(1e-5);
      expect(relativeError(0.75, n)).toBeGreaterThan(0.39);
    }
  });
});
