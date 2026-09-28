"use client";

import { useState } from "react";

// The two TD updates checked in tests/test_tabular_q_learning_agent.cpp:37-74
// (terminated, expects 2.5 at :73) and :76-112 (truncated, expects 25 at :111): learning_rate 0.5, discount 0.9,
// Q(s1, 0) pre-seeded to 50, then a transition s0 --a0, r=5--> s1.
const LR = 0.5; // tests/test_tabular_q_learning_agent.cpp:40
const GAMMA = 0.9; // tests/test_tabular_q_learning_agent.cpp:41
const R = 5; // tests/test_tabular_q_learning_agent.cpp:66 and :102
const Q_NEXT = 50; // tests/test_tabular_q_learning_agent.cpp:58 — REQUIRE(q_value(state1, 0) == Approx(50.0f))
const Q_OLD = 0;

type Case = "terminated" | "truncated" | "done";

const CASES: { id: Case; label: string; sub: string }[] = [
  { id: "terminated", label: "terminated", sub: "the MDP really ended" },
  { id: "truncated", label: "truncated", sub: "a time limit cut it off" },
  { id: "done", label: "one `done` flag", sub: "truncated, collapsed" },
];

export function BellmanToggle() {
  const [c, setC] = useState<Case>("truncated");
  // "done" is the classic bug: a truncated transition treated as terminal.
  const bootstrap = c === "truncated";
  const mask = bootstrap ? 1 : 0;
  const target = R + GAMMA * mask * Q_NEXT;
  const q = Q_OLD + LR * (target - Q_OLD);
  const correct = c === "done" ? R + GAMMA * Q_NEXT : target;
  const qCorrect = Q_OLD + LR * (correct - Q_OLD);
  const wrong = c === "done";

  return (
    <div className="card spot flex h-full flex-col p-5 sm:p-6">
      <div role="radiogroup" aria-label="How the transition ended" className="grid grid-cols-3 gap-1 rounded-2xl border border-line bg-bg-2 p-1">
        {CASES.map((k) => (
          <button
            key={k.id}
            type="button"
            role="radio"
            aria-checked={c === k.id}
            onClick={() => setC(k.id)}
            className={`min-w-0 rounded-xl px-2 py-2 text-left transition-all duration-300 ease-spring sm:px-3 ${
              c === k.id ? (k.id === "done" ? "bg-bad/10 ring-1 ring-bad/50" : "bg-amber/10 ring-1 ring-amber/50") : "hover:bg-surface-2"
            }`}
          >
            <span className={`block truncate font-mono text-[0.78rem] font-semibold ${c === k.id ? (k.id === "done" ? "text-bad" : "text-amber") : "text-ink"}`}>
              {k.label.replace(/`/g, "")}
            </span>
            <span className="block truncate text-[0.68rem] text-ink-3">{k.sub}</span>
          </button>
        ))}
      </div>

      {/* the transition */}
      <div className="mt-6 flex items-center justify-between gap-2 font-mono text-sm">
        <span className="rounded-xl border border-line-2 bg-surface-2 px-3 py-2 text-ink">s₀</span>
        <span className="flex-1 text-center text-[0.75rem] text-ink-3">
          a₀, r = {R}
          <span className="mx-auto mt-1 block h-px w-full bg-gradient-to-r from-line-2 via-ink-3 to-line-2" />
        </span>
        <span
          className={`rounded-xl border px-3 py-2 transition-colors duration-300 ${
            bootstrap ? "border-steel/60 bg-steel/10 text-steel" : "border-line-2 bg-surface-2 text-ink-3"
          }`}
        >
          s₁ · max Q = {Q_NEXT}
        </span>
      </div>

      {/* the target */}
      <div className="mt-6 rounded-2xl border border-line bg-bg-2 px-4 py-4 font-mono text-[0.8rem] leading-loose text-ink-2 sm:text-[0.9rem]">
        <div>
          y = r + γ · <span className={bootstrap ? "text-steel" : "text-bad"}>(1 − terminated)</span> · maxₐ Q(s′, a)
        </div>
        <div>
          y = {R} + {GAMMA} · <span className={bootstrap ? "text-steel" : "text-bad"}>{mask}</span> ·{" "}
          <span className={`transition-all duration-300 ${bootstrap ? "text-steel" : "text-ink-3 line-through decoration-bad"}`}>{Q_NEXT}</span> ={" "}
          <span className="text-ink">{target}</span>
        </div>
        <div>
          Q(s₀, a₀) ← {Q_OLD} + {LR} · ({target} − {Q_OLD}) ={" "}
          <span className={`text-lg font-semibold ${wrong ? "text-bad" : "text-hot"}`}>{q}</span>
        </div>
      </div>

      <p className="mt-4 min-h-[4.5rem] text-sm leading-relaxed text-ink-2" aria-live="polite">
        {c === "terminated" && <>No next state exists, so the bootstrap term is zeroed. The test expects exactly 2.5.</>}
        {c === "truncated" && <>The world didn&apos;t end, the clock did. s₁ is real, so its value stays in the target. The test expects exactly 25.</>}
        {c === "done" && (
          <>
            Collapse both into <code className="font-mono text-ink">done</code> and this time-limit transition gets {q} instead of {qCorrect}. Your
            agent learns a world where every clock runs out into nothing.
          </>
        )}
      </p>
    </div>
  );
}
