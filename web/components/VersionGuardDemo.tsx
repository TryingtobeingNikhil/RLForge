"use client";

import { useState } from "react";
import { VG_CORRECT_GRAD, VG_MUTATION, VG_STALE_GRAD, VG_THROW_MESSAGE, VG_X } from "@/lib/versionGuard";
import { PlayIcon, ResetIcon, StepIcon } from "./Icons";

// tests/test_version_guard.cpp:30-41, one line per step.
const LINES = [
  { code: "auto x = Tensor::from_data({1.0, 2.0, 3.0}, {3});", line: 30 },
  { code: "x.requires_grad_(true);", line: 31 },
  { code: "auto loss = x.mul(x).mean();", line: 33 },
  { code: "auto buf = x.data_mutable();", line: 37 },
  { code: "buf[0] = 99.0;  // mutation", line: 38 },
  { code: "loss.backward();", line: 41 },
];

// The frames a throw from check_version() passes through, read from the
// source (not a captured runtime backtrace).
const THROW_PATH = [
  { fn: "check_version(storage, saved_version, \"mul\", \"rhs\")", at: "src/tensor/tensor.cpp:72" },
  { fn: "mul::backward_fn(grad)", at: "src/tensor/tensor.cpp:525" },
  { fn: "run_backward(root_node, seed)", at: "src/tensor/tensor.cpp:156" },
  { fn: "Tensor::backward()", at: "src/tensor/tensor.cpp:316" },
  { fn: "your training step", at: "tests/test_version_guard.cpp:41" },
];

const fmt = (v: number) => (Number.isInteger(v) ? v.toFixed(1) : v.toFixed(3));

export function VersionGuardDemo() {
  const [ran, setRan] = useState(0); // lines executed
  const [showUnguarded, setShowUnguarded] = useState(false);

  const data = ran >= 5 ? VG_X.map((v, i) => (i === VG_MUTATION.index ? VG_MUTATION.value : v)) : VG_X;
  const version = ran >= 5 ? 1 : 0;
  const graphBuilt = ran >= 3;
  const threw = ran >= 6;
  const allocated = ran >= 1;

  return (
    <div className="card spot flex h-full flex-col p-5 sm:p-6">
      <div className="flex flex-wrap gap-2">
        <button type="button" className="ctl" disabled={threw} onClick={() => setRan((r) => Math.min(6, r + 1))}>
          <StepIcon className="h-3.5 w-3.5" /> Run next line
        </button>
        <button type="button" className="ctl" disabled={threw} onClick={() => setRan(6)}>
          <PlayIcon className="h-3.5 w-3.5" /> Run all
        </button>
        <button type="button" className="ctl" disabled={ran === 0} onClick={() => { setRan(0); setShowUnguarded(false); }}>
          <ResetIcon className="h-3.5 w-3.5" /> Reset
        </button>
      </div>

      <ol className="mt-5 overflow-x-auto rounded-2xl border border-line bg-[#0e0f12] py-2 font-mono text-[0.8rem]">
        {LINES.map((l, i) => {
          const state = i < ran ? "done" : i === ran ? "next" : "todo";
          const isThrowLine = i === 5 && threw;
          return (
            <li
              key={l.line}
              className={`flex items-center gap-3 whitespace-nowrap px-4 py-1 transition-colors duration-300 ${
                isThrowLine ? "bg-bad/10" : state === "next" ? "bg-amber/[0.07]" : ""
              }`}
            >
              <span className="w-6 shrink-0 text-right text-[0.7rem] text-ink-3">{l.line}</span>
              <span className={`w-3 shrink-0 ${isThrowLine ? "text-bad" : state === "next" ? "text-amber" : "text-ok"}`} aria-hidden>
                {isThrowLine ? "✕" : state === "done" ? "✓" : state === "next" ? "›" : ""}
              </span>
              <span className={state === "todo" ? "text-ink-3" : isThrowLine ? "text-bad" : "text-ink"}>{l.code}</span>
            </li>
          );
        })}
      </ol>

      {/* storage + closure state */}
      <div className="mt-4 grid grid-cols-2 gap-3 font-mono text-[0.78rem]">
        <div className="rounded-xl border border-line bg-bg-2 p-3">
          <div className="text-[0.66rem] uppercase tracking-[0.14em] text-ink-3">x.storage</div>
          <div className="mt-1.5 text-ink">{allocated ? `[${data.map((v) => v.toFixed(1)).join(", ")}]` : "·"}</div>
          <div className="mt-1 text-ink-2">
            version{" "}
            <span key={version} className={`inline-block transition-transform duration-500 ease-spring ${version ? "scale-110 text-amber" : "text-ink"}`}>
              {allocated ? version : "·"}
            </span>
          </div>
        </div>
        <div className="rounded-xl border border-line bg-bg-2 p-3">
          <div className="text-[0.66rem] uppercase tracking-[0.14em] text-ink-3">mul backward closure</div>
          <div className="mt-1.5 text-ink">{graphBuilt ? "captured at forward" : "·"}</div>
          <div className="mt-1 text-ink-2">
            saved version <span className="text-ink">{graphBuilt ? 0 : "·"}</span>
            {graphBuilt && version !== 0 ? <span className="ml-1 text-bad">≠ {version}</span> : null}
          </div>
        </div>
      </div>

      {threw ? (
        <div className="mt-4 overflow-hidden rounded-2xl border border-bad/50 bg-[#1a0f0f]" role="alert">
          <div className="flex items-center justify-between gap-2 border-b border-bad/30 px-4 py-2">
            <span className="font-mono text-[0.78rem] font-semibold text-bad">throws std::runtime_error</span>
            <span className="font-mono text-[0.66rem] text-ink-3">instead of a bad gradient</span>
          </div>
          <p className="break-words px-4 pt-3 font-mono text-[0.76rem] leading-relaxed text-[#ffc9c2]">what(): {VG_THROW_MESSAGE}</p>
          <ol className="px-4 pb-3 pt-2 font-mono text-[0.72rem] leading-relaxed">
            {THROW_PATH.map((f, i) => (
              <li key={f.at} className="flex min-w-0 flex-wrap gap-x-2">
                <span className="text-ink-3">#{i}</span>
                <span className="text-ink-2">{f.fn}</span>
                <span className="text-ink-3">{f.at}</span>
              </li>
            ))}
          </ol>
          <p className="border-t border-bad/20 px-4 py-2 text-[0.72rem] text-ink-3">
            Frames read from the source, not captured at runtime. The message is built exactly as check_version() builds it.
          </p>
        </div>
      ) : (
        <p className="mt-4 text-sm leading-relaxed text-ink-2">
          {ran < 3 && "Build the graph first. The closure for x·x records the storage version it saw."}
          {ran >= 3 && ran < 5 && "Now mutate x in place, the way an optimizer step would."}
          {ran === 5 && "Storage says version 1. The closure remembers 0. Call backward()."}
        </p>
      )}

      {threw ? (
        <div className="mt-4">
          <button type="button" className="text-left text-sm text-ink-2 underline decoration-line-2 underline-offset-4 hover:decoration-amber" aria-expanded={showUnguarded} onClick={() => setShowUnguarded((v) => !v)}>
            {showUnguarded ? "Hide" : "What would an engine without the guard return?"}
          </button>
          {showUnguarded ? (
            <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-[0.8rem]">
              <dt className="text-ink-3">matches the forward pass</dt>
              <dd className="text-ok">[{VG_CORRECT_GRAD.map(fmt).join(", ")}]</dd>
              <dt className="text-ink-3">silently returned</dt>
              <dd className="text-bad">[{VG_STALE_GRAD.map(fmt).join(", ")}]</dd>
            </dl>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
