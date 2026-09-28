"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { backward, DIAMOND_X, forward, numericalGradient, relativeError, type Mode, type NodeId } from "@/lib/diamond";
import { useInView } from "@/hooks/useInView";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { PauseIcon, PlayIcon, ResetIcon, StepIcon } from "./Icons";

const POS: Record<NodeId, { x: number; y: number }> = {
  x: { x: 210, y: 52 },
  a: { x: 92, y: 196 },
  b: { x: 328, y: 196 },
  c: { x: 210, y: 340 },
  loss: { x: 210, y: 470 },
};
// Node boxes extend TOP above and BOTTOM below their center.
const TOP = 28;
const BOTTOM = 40;
const FORWARD_EDGES: { from: NodeId; to: NodeId; op: string }[] = [
  { from: "x", to: "a", op: "×2" },
  { from: "x", to: "b", op: "×3" },
  { from: "a", to: "c", op: "+" },
  { from: "b", to: "c", op: "+" },
  { from: "c", to: "loss", op: "mean" },
];
const LABEL: Record<NodeId, string> = { x: "x", a: "a = 2x", b: "b = 3x", c: "c = a + b", loss: "loss = mean(c)" };

const TRAVEL_MS = 700;
const HOLD_MS = 2400;

const fmt = (v: number) => (Number.isInteger(v) ? v.toFixed(0) : v.toFixed(2).replace(/0$/, ""));
const vec = (v: number[]) => `[${v.map(fmt).join(", ")}]`;

export function DiamondBug() {
  const [mode, setMode] = useState<Mode>("accumulate");
  const [stage, setStage] = useState(0); // deposits fully landed
  const [progress, setProgress] = useState(0); // 0..1 of the in-flight deposit
  const [playing, setPlaying] = useState(true);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const reduced = useReducedMotion();

  const f = useMemo(() => forward(), []);
  const trace = useMemo(() => backward(mode), [mode]);
  const numeric = useMemo(() => numericalGradient(), []);
  const total = trace.deposits.length;

  // Reduced motion: no traveling particles, show the finished backward pass.
  useEffect(() => {
    if (reduced) {
      setPlaying(false);
      setStage(total);
      setProgress(0);
    }
  }, [reduced, total]);

  useEffect(() => {
    if (!playing || !inView || reduced) return;
    let raf = 0;
    let last = performance.now();
    let hold = 0;
    let s = stage;
    let p = progress;
    const tick = (now: number) => {
      const dt = Math.min(now - last, 64);
      last = now;
      if (s >= total) {
        hold += dt;
        if (hold >= HOLD_MS) {
          hold = 0;
          s = 0;
          p = 0;
          setStage(0);
          setProgress(0);
        }
      } else {
        p += dt / TRAVEL_MS;
        if (p >= 1) {
          s += 1;
          p = 0;
          setStage(s);
        }
        setProgress(p);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the loop owns stage/progress while running
  }, [playing, inView, reduced, total, mode]);

  const switchMode = (m: Mode) => {
    if (m === mode) return;
    setMode(m);
    setStage(reduced ? total : 0);
    setProgress(0);
    if (!reduced) setPlaying(true);
  };

  const gradAt = (node: NodeId): number[] | null => {
    if (node === "loss") return [1];
    let g: number[] | null = null;
    for (let i = 0; i < stage; i++) if (trace.deposits[i].to === node) g = trace.deposits[i].after;
    return g;
  };

  const inFlight = stage < total && (playing || progress > 0) && !reduced ? trace.deposits[stage] : null;
  const intoX = trace.deposits.filter((d) => d.to === "x");
  const xLanded = trace.deposits.slice(0, stage).filter((d) => d.to === "x").length;
  const finalX = trace.grads.x![0];
  const numericX = numeric[0];
  const relErr = relativeError(finalX, numericX);
  const pass = relErr < 1e-5;
  const done = stage >= total;

  return (
    <div ref={ref} className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" data-offscreen={!inView}>
      {/* graph */}
      <div className="card spot overflow-hidden p-3 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 px-1 pb-2">
          <span className="font-mono text-xs text-ink-3">x = {vec(DIAMOND_X)} · backward pass</span>
          <span className={`font-mono text-xs ${mode === "accumulate" ? "text-ok" : "text-bad"}`}>
            {mode === "accumulate" ? "grad += contribution" : "grad = contribution"}
          </span>
        </div>
        <svg viewBox="0 0 420 520" className="mx-auto block w-full max-w-[30rem]" role="img" aria-label={`Diamond graph. With ${mode}, x.grad ends at ${fmt(finalX)} per element.`}>
          <defs>
            <marker id="dm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10z" fill="rgb(var(--line-2))" />
            </marker>
            <filter id="dm-glow" x="-100%" y="-100%" width="300%" height="300%">
              <feGaussianBlur stdDeviation="4" />
            </filter>
          </defs>

          {FORWARD_EDGES.map((e) => {
            const A = POS[e.from];
            const B = POS[e.to];
            const active = inFlight && inFlight.from === e.to && inFlight.to === e.from;
            const landed = trace.deposits.slice(0, stage).some((d) => d.from === e.to && d.to === e.from);
            const mx = (A.x + B.x) / 2;
            const my = (A.y + B.y) / 2;
            return (
              <g key={`${e.from}-${e.to}`}>
                <line
                  x1={A.x}
                  y1={A.y + BOTTOM}
                  x2={B.x}
                  y2={B.y - TOP - 2}
                  stroke={active || landed ? "rgb(var(--steel) / 0.7)" : "rgb(var(--line-2))"}
                  strokeWidth={active ? 3 : 2}
                  markerEnd="url(#dm-arrow)"
                />
                <rect x={mx - 20} y={my - 12} width={40} height={22} rx={11} fill="rgb(var(--bg-2))" stroke="rgb(var(--line))" />
                <text x={mx} y={my + 4} textAnchor="middle" fontSize="12" fill="rgb(var(--ink-2))" className="font-mono">
                  {e.op}
                </text>
              </g>
            );
          })}

          {inFlight ? (
            <g pointerEvents="none">
              {(() => {
                const A = POS[inFlight.from];
                const B = POS[inFlight.to];
                const e = 1 - Math.pow(1 - progress, 3);
                const px = A.x + (B.x - A.x) * e;
                const py = A.y - TOP + (B.y + BOTTOM - (A.y - TOP)) * e;
                return (
                  <>
                    <circle cx={px} cy={py} r={13} fill="rgb(var(--steel))" opacity={0.55} filter="url(#dm-glow)" />
                    <circle cx={px} cy={py} r={6} fill="rgb(var(--steel))" />
                    <text x={px + 14} y={py - 10} fontSize="13" fill="rgb(var(--steel))" className="font-mono">
                      +{fmt(inFlight.contribution[0])}
                    </text>
                  </>
                );
              })()}
            </g>
          ) : null}

          {(Object.keys(POS) as NodeId[]).map((id) => {
            const p = POS[id];
            const g = gradAt(id);
            const isX = id === "x";
            const wrong = isX && done && mode === "overwrite";
            const right = isX && done && mode === "accumulate";
            const w = id === "loss" ? 150 : id === "c" ? 136 : 128;
            return (
              <g key={id}>
                <rect
                  x={p.x - w / 2}
                  y={p.y - TOP}
                  width={w}
                  height={TOP + BOTTOM}
                  rx={16}
                  fill="rgb(var(--surface-2))"
                  stroke={wrong ? "rgb(var(--bad))" : right ? "rgb(var(--ok))" : isX ? "rgb(var(--amber) / 0.7)" : "rgb(var(--line-2))"}
                  strokeWidth={isX ? 2 : 1.25}
                />
                <text x={p.x} y={p.y - 8} textAnchor="middle" fontSize="14" fontWeight="600" fill="rgb(var(--ink))">
                  {LABEL[id]}
                </text>
                <text x={p.x} y={p.y + 9} textAnchor="middle" fontSize="11" fill="rgb(var(--ink-3))" className="font-mono">
                  {id === "loss" ? fmt(f.loss) : vec(f[id])}
                </text>
                <text
                  x={p.x}
                  y={p.y + 28}
                  textAnchor="middle"
                  fontSize="12"
                  className="font-mono"
                  fill={wrong ? "rgb(var(--bad))" : right ? "rgb(var(--ok))" : g ? "rgb(var(--steel))" : "rgb(var(--ink-3))"}
                >
                  {g ? `grad ${fmt(g[0])}${g.length > 1 ? " each" : ""}` : "grad ·"}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* controls + ledger */}
      <div className="flex min-w-0 flex-col gap-4">
        <div role="radiogroup" aria-label="Gradient deposit rule" className="grid grid-cols-2 gap-1 rounded-2xl border border-line bg-bg-2 p-1">
          {(
            [
              ["accumulate", "Accumulate", "what RLForge does"],
              ["overwrite", "Overwrite", "the bug"],
            ] as const
          ).map(([m, label, sub]) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              onClick={() => switchMode(m)}
              className={`rounded-xl px-3 py-2.5 text-left transition-all duration-300 ease-spring ${
                mode === m
                  ? m === "accumulate"
                    ? "bg-ok/10 ring-1 ring-ok/50"
                    : "bg-bad/10 ring-1 ring-bad/50"
                  : "hover:bg-surface-2"
              }`}
            >
              <span className={`block text-sm font-semibold ${mode === m ? (m === "accumulate" ? "text-ok" : "text-bad") : "text-ink"}`}>{label}</span>
              <span className="block font-mono text-[0.7rem] text-ink-3">{sub}</span>
            </button>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          <button type="button" className="ctl" onClick={() => setPlaying((p) => !p)} disabled={reduced} aria-pressed={playing}>
            {playing ? <PauseIcon className="h-3.5 w-3.5" /> : <PlayIcon className="h-3.5 w-3.5" />} {playing ? "Pause" : "Play"}
          </button>
          <button
            type="button"
            className="ctl"
            onClick={() => {
              setPlaying(false);
              setProgress(0);
              setStage((s) => (s >= total ? 0 : s + 1));
            }}
          >
            <StepIcon className="h-3.5 w-3.5" /> {done ? "Start over" : "Next deposit"}
          </button>
          <button
            type="button"
            className="ctl"
            onClick={() => {
              setStage(0);
              setProgress(0);
              if (!reduced) setPlaying(true);
            }}
          >
            <ResetIcon className="h-3.5 w-3.5" /> Replay
          </button>
        </div>

        <ol className="card divide-y divide-line overflow-hidden" aria-label="Gradient deposits, in backward order">
          {trace.deposits.map((d, i) => {
            const landed = i < stage;
            const isX = d.to === "x";
            const overwrote = mode === "overwrite" && isX && d.before;
            return (
              <li
                key={d.edge}
                className={`flex items-center justify-between gap-3 px-4 py-2.5 font-mono text-[0.8rem] transition-opacity duration-300 ${landed ? "opacity-100" : "opacity-40"}`}
              >
                <span className="text-ink-2">
                  {d.from === "loss" ? "loss" : d.from} <span className="text-ink-3">→</span> {d.to}
                </span>
                <span className="text-right tabular-nums">
                  {overwrote ? (
                    <>
                      <span className="text-ink-3 line-through decoration-bad">{fmt(d.before![0])}</span>{" "}
                      <span className="text-bad">= {fmt(d.after[0])}</span>
                    </>
                  ) : d.before ? (
                    <span className="text-ok">
                      {fmt(d.before[0])} + {fmt(d.contribution[0])} = {fmt(d.after[0])}
                    </span>
                  ) : (
                    <span className="text-steel">{fmt(d.after[0])}</span>
                  )}
                </span>
              </li>
            );
          })}
        </ol>

        <div className={`card p-4 transition-colors duration-500 ${done ? (pass ? "border-ok/40" : "border-bad/50") : ""}`}>
          <p className="font-mono text-[0.7rem] uppercase tracking-[0.14em] text-ink-3">check_grad, per element of x</p>
          <dl className="mt-3 grid grid-cols-3 gap-3 font-mono">
            <div>
              <dt className="text-[0.68rem] text-ink-3">analytic</dt>
              <dd className={`text-lg tabular-nums ${!done ? "text-ink-3" : pass ? "text-ok" : "text-bad"}`}>{done ? fmt(finalX) : "…"}</dd>
            </div>
            <div>
              <dt className="text-[0.68rem] text-ink-3">numerical (ε=1e-5)</dt>
              <dd className="text-lg tabular-nums text-steel">{numericX.toFixed(6)}</dd>
            </div>
            <div>
              <dt className="text-[0.68rem] text-ink-3">rel. error</dt>
              <dd className={`text-lg tabular-nums ${!done ? "text-ink-3" : pass ? "text-ok" : "text-bad"}`}>{done ? relErr.toExponential(1) : "…"}</dd>
            </div>
          </dl>
          <p className="mt-3 text-sm leading-relaxed text-ink-2" aria-live="polite">
            {!done
              ? `Gradients are still flowing back toward x (${xLanded} of ${intoX.length} deposits in).`
              : pass
                ? "Both paths summed: 2/4 + 3/4 = 1.25. Within the suite's 1e-5 tolerance. Test passes."
                : "b's deposit replaced a's. 0.75 is off by 40% and nothing crashed: this is the silent failure. check_grad would fail it."}
          </p>
        </div>
      </div>
    </div>
  );
}
