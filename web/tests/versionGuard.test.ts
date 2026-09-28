import { describe, expect, it } from "vitest";
import { VG_CORRECT_GRAD, VG_STALE_GRAD, VG_THROW_MESSAGE } from "@/lib/versionGuard";

describe("version guard model", () => {
  it("reproduces the words tests/test_version_guard.cpp:57-58 look for", () => {
    expect(VG_THROW_MESSAGE).toContain("backward");
    expect(VG_THROW_MESSAGE).toContain("mutated");
    expect(VG_THROW_MESSAGE).toContain("(saved version=0, current version=1)");
  });

  it("shows how wrong the unguarded gradient would be", () => {
    expect(VG_CORRECT_GRAD.map((v) => +v.toFixed(4))).toEqual([0.6667, 1.3333, 2]);
    expect(VG_STALE_GRAD[0]).toBe(66);
  });
});
