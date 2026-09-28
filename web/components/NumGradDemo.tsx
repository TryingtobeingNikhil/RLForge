"use client";

import { useState } from "react";
import { NUMGRAD_EPS, relativeError } from "@/lib/diamond";

// The square op's three checks: forward x^2 (tests/test_tensor.cpp:221),
// analytic backward 2x (:445), numerical central difference (:717), with the
// suite's eps = 1e-5 and relative tolerance 1e-5 (tests/test_tensor.cpp:33-34).
const TOL = 1e-5;
const f = (x: number) => x * x;

export function NumGradDemo() {
  const [x, setX] = useState(1.5);
  const [buggy, setBuggy] = useState(false);
  const analytic = buggy ? x : 2 * x; // the "bug": forgetting the factor 2
  const numeric = (f(x + NUMGRAD_EPS) - f(x - NUMGRAD_EPS)) / (2 * NUMGRAD_EPS);
  const err = relativeError(analytic, numeric);
  const pass = err < TOL;

  return (
    <div className="card spot p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="font-mono text-sm text-ink">
          square(x) at x = <span className="tabular-nums text-amber">{x.toFixed(2)}</span>
        </p>
        <label className="flex cursor-pointer items-center gap-2 text-[0.8rem] text-ink-2">
          <input type="checkbox" checked={buggy} onChange={(e) => setBuggy(e.target.checked)} className="h-4 w-4 accent-[rgb(var(--bad))]" />
          plant a bug: backward returns x
        </label>
      </div>
      <input
        type="range"
        min={-3}
        max={3}
        step={0.01}
        value={x}
        onChange={(e) => setX(parseFloat(e.target.value))}
        aria-label="x"
        className="mt-4 w-full accent-[rgb(var(--amber))]"
      />
      <dl className="mt-4 grid grid-cols-2 gap-3 font-mono sm:grid-cols-4">
        <div>
          <dt className="text-[0.66rem] uppercase tracking-[0.12em] text-ink-3">forward</dt>
          <dd className="mt-1 tabular-nums text-ink">{f(x).toFixed(4)}</dd>
        </div>
        <div>
          <dt className="text-[0.66rem] uppercase tracking-[0.12em] text-ink-3">analytic</dt>
          <dd className={`mt-1 tabular-nums ${buggy ? "text-bad" : "text-ink"}`}>{analytic.toFixed(6)}</dd>
        </div>
        <div>
          <dt className="text-[0.66rem] uppercase tracking-[0.12em] text-ink-3">numerical</dt>
          <dd className="mt-1 tabular-nums text-steel">{numeric.toFixed(6)}</dd>
        </div>
        <div>
          <dt className="text-[0.66rem] uppercase tracking-[0.12em] text-ink-3">rel. error</dt>
          <dd className={`mt-1 tabular-nums ${pass ? "text-ok" : "text-bad"}`}>{err.toExponential(1)}</dd>
        </div>
      </dl>
      <p className={`mt-4 text-sm ${pass ? "text-ok" : "text-bad"}`} aria-live="polite">
        {pass ? "✓ within 1e-5: the analytic gradient checks out numerically." : "✕ over 1e-5: the test fails, and the bug never reaches a training run."}
      </p>
      <p className="mt-1 text-[0.72rem] text-ink-3">
        Computed live in double precision with the suite&apos;s central difference: (f(x+ε) − f(x−ε)) / 2ε, ε = 1e-5.
      </p>
    </div>
  );
}
