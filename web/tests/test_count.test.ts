import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { TEST_GROUPS, TEST_TAGS, TOTAL_TESTS } from "@/lib/content";

// Re-counts TEST_CASEs in the C++ suite so the site's numbers can't drift.
// Skips when the C++ sources aren't next to web/ (e.g. a web-only checkout).
const repoRoot = path.resolve(__dirname, "../..");
const testsDir = path.join(repoRoot, "tests");
const hasCppTests = fs.existsSync(path.join(testsDir, "CMakeLists.txt"));

// The test files committed to git. A local, uncommitted test file is not
// part of the published suite, so it doesn't count.
function committedTestFiles(): string[] {
  const out = execFileSync("git", ["ls-files", "tests"], { cwd: repoRoot, encoding: "utf8" });
  return out.split("\n").filter((f) => /^tests\/test_\w+\.cpp$/.test(f));
}

function parse(file: string) {
  const src = fs.readFileSync(path.join(repoRoot, file), "utf8");
  const cases = [...src.matchAll(/TEST_CASE\s*\(\s*"(?:[^"\\]|\\.)*"\s*,\s*"([^"]*)"/g)];
  return { count: cases.length, tags: cases.flatMap((m) => [...m[1].matchAll(/\[([^\]]+)\]/g)].map((t) => t[1])) };
}

describe.skipIf(!hasCppTests)("test counts match the C++ sources", () => {
  const siteFiles = TEST_GROUPS.flatMap((g) => g.files);

  it("totals 148", () => {
    expect(TOTAL_TESTS).toBe(148);
  });

  it("per-file counts match every TEST_CASE in the committed suite", () => {
    for (const { file, count } of siteFiles) expect(parse(file).count, file).toBe(count);
  });

  it("covers exactly the committed test files", () => {
    expect(siteFiles.map((f) => f.file).sort()).toEqual(committedTestFiles().sort());
  });

  it("tag counts match", () => {
    const counts = new Map<string, number>();
    for (const { file } of siteFiles) for (const t of parse(file).tags) counts.set(t, (counts.get(t) ?? 0) + 1);
    expect(Object.fromEntries(counts)).toEqual(Object.fromEntries(TEST_TAGS.map((t) => [t.tag, t.count])));
  });
});
