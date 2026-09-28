import { describe, expect, it } from "vitest";
import { MT19937, uniformFloat01, uniformIntBelow } from "@/lib/mt19937";

describe("MT19937 port", () => {
  it("matches the C++ standard's check value: the 10000th output of a default-seeded mt19937 is 4123659995", () => {
    // [rand.predef] in the C++ standard pins this value for std::mt19937.
    const rng = new MT19937();
    let v = 0;
    for (let i = 0; i < 10000; i++) v = rng.nextU32();
    expect(v).toBe(4123659995);
  });

  it("keeps distributions in range", () => {
    const rng = new MT19937(0);
    for (let i = 0; i < 10000; i++) {
      const u = uniformFloat01(rng);
      expect(u).toBeGreaterThanOrEqual(0);
      expect(u).toBeLessThan(1);
      const a = uniformIntBelow(rng, 4);
      expect([0, 1, 2, 3]).toContain(a);
    }
  });
});
