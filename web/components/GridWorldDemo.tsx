"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ACTION_NAMES, GRIDWORLD_CONFIG as C, GridWorldTrainer, type Obs } from "@/lib/gridworld";
import { useInView } from "@/hooks/useInView";
import { useReducedMotion } from "@/hooks/useReducedMotion";
import { PauseIcon, PlayIcon, ResetIcon, StepIcon } from "./Icons";

const SIZE = C.size;
const CELL = 100;
const GOAL: Obs = { x: SIZE - 1, y: SIZE - 1 };

// Trainer steps per second. Each trainer step moves all 4 lanes once.
const SPEEDS = [
  { id: "1x", label: "1×", sps: 5 },
  { id: "10x", label: "10×", sps: 50 },
  { id: "100x", label: "100×", sps: 500 },
  { id: "max", label: "Max", sps: 5000 },
] as const;
type SpeedId = (typeof SPEEDS)[number]["id"];
const MAX_STEPS_PER_FRAME = 400;

interface EvalPoint {
  step: number;
  value: number;
}

// Heat ramp for V(s) = max_a Q(s, a): cold steel below zero, glowing iron to
// white-hot as the value approaches the +10 goal reward.
const RAMP: [number, [number, number, number]][] = [
  [-5, [30, 50, 78]],
  [-1.5, [32, 40, 52]],
  [0, [34, 37, 44]],
  [2.5, [84, 38, 24]],
  [5, [156, 54, 22]],
  [7.5, [236, 102, 36]],
  [10, [255, 200, 118]],
];
function heat(v: number): string {
  const x = Math.max(RAMP[0][0], Math.min(RAMP[RAMP.length - 1][0], v));
  for (let i = 1; i < RAMP.length; i++) {
    const [v1, c1] = RAMP[i];
    const [v0, c0] = RAMP[i - 1];
    if (x <= v1) {
      const t = (x - v0) / (v1 - v0);
      const c = c0.map((ch, k) => Math.round(ch + (c1[k] - ch) * t));
      return `rgb(${c[0]} ${c[1]} ${c[2]})`;
    }
  }
  return "rgb(255 200 118)";
}

const cx = (x: number) => x * CELL + CELL / 2;
const cy = (y: number) => (SIZE - 1 - y) * CELL + CELL / 2; // y grows upward: "Up" is y + 1

// Arrow directions in SVG space for Up, Down, Left, Right.
const DIRS: [number, number][] = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];
function arrowPath(X: number, Y: number, dx: number, dy: number): string {
  const tx = X + dx * 17;
  const ty = Y + dy * 17;
  const w1 = [tx - dx * 10 - dy * 9, ty - dy * 10 + dx * 9];
  const w2 = [tx - dx * 10 + dy * 9, ty - dy * 10 - dx * 9];
  return `M${X - dx * 17} ${Y - dy * 17} L${tx} ${ty} M${w1[0]} ${w1[1]} L${tx} ${ty} L${w2[0]} ${w2[1]}`;
}

// Lanes sit in a row along the bottom of a cell; V(s) is printed top-left.
const LANE_OFFSETS: [number, number][] = [
  [-30, 31],
  [-10, 31],
  [10, 31],
  [30, 31],
];

const fmtInt = new Intl.NumberFormat("en");
const fmtRet = (v: number) => (v < 0 ? `−${Math.abs(v).toFixed(1)}` : v.toFixed(1));
const fmtQ = (v: number) => (Math.abs(v) < 0.005 ? "0.00" : v < 0 ? `−${Math.abs(v).toFixed(2)}` : v.toFixed(2));

// Chart: symmetric-log x so the first few hundred steps, where the policy
// actually changes, are as readable as the long converged tail.
// The chart is drawn in real pixels: its viewBox width tracks the rendered
// width (see chartWidth below), so labels stay legible on phones.
const CH = { h: 226, l: 46, r: 16, t: 18, b: 46 };
const LOG_MAX = Math.log10(1 + C.trainSteps);
// Rounded so server and browser Math.log10 (which can differ in the last
// bit) render identical attributes during hydration.
const r2 = (v: number) => Math.round(v * 100) / 100;
const makeChartX = (w: number) => (step: number) => r2(CH.l + (Math.log10(1 + step) / LOG_MAX) * (w - CH.l - CH.r));
const Y_MIN = -100;
const Y_MAX = 10;
const chartY = (v: number) => r2(CH.t + ((Y_MAX - v) / (Y_MAX - Y_MIN)) * (CH.h - CH.t - CH.b));
const X_TICKS = [
  [0, "0"],
  [10, "10"],
  [100, "100"],
  [1000, "1k"],
  [10000, "10k"],
  [20000, "20k"],
] as const;

export function GridWorldDemo() {
  const trainerRef = useRef<GridWorldTrainer>();
  const historyRef = useRef<EvalPoint[]>([]);
  const firstOptimalRef = useRef<number | null>(null);
  const [, bump] = useReducer((n: number) => n + 1, 0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<SpeedId>("10x");
  const [selected, setSelected] = useState<Obs>({ x: 0, y: 0 });
  const [announce, setAnnounce] = useState("");
  const autoplayedRef = useRef(false);

  const rootRef = useRef<HTMLDivElement>(null);
  const inView = useInView(rootRef, "80px");
  const reduced = useReducedMotion();

  const figRef = useRef<HTMLElement>(null);
  const [chartWidth, setChartWidth] = useState(600);
  useEffect(() => {
    const el = figRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setChartWidth(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const chartX = makeChartX(chartWidth);

  if (!trainerRef.current) {
    trainerRef.current = new GridWorldTrainer();
    historyRef.current = [{ step: 0, value: trainerRef.current.evaluate() }];
  }
  const trainer = trainerRef.current;

  const advance = useCallback((n: number) => {
    const t = trainerRef.current!;
    const hist = historyRef.current;
    for (let i = 0; i < n && !t.done; i++) {
      t.step();
      const value = t.evaluate();
      if (value !== hist[hist.length - 1].value) hist.push({ step: t.trainStep, value });
      if (value === 3 && firstOptimalRef.current === null) firstOptimalRef.current = t.trainStep;
    }
  }, []);

  // Autoplay once, the first time the demo scrolls into view, unless the
  // visitor prefers reduced motion (then it waits for a click on Play).
  useEffect(() => {
    if (inView && !autoplayedRef.current && !reduced) {
      autoplayedRef.current = true;
      setPlaying(true);
    }
  }, [inView, reduced]);

  const done = trainer.done;
  const running = playing && inView && !done;

  useEffect(() => {
    if (!running) return;
    const sps = SPEEDS.find((s) => s.id === speed)!.sps;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      acc += (sps * Math.min(now - last, 100)) / 1000;
      last = now;
      const n = Math.min(Math.floor(acc), MAX_STEPS_PER_FRAME);
      if (n > 0) {
        acc -= n;
        advance(n);
        bump();
      }
      if (!trainerRef.current!.done) raf = requestAnimationFrame(tick);
      else {
        setPlaying(false);
        setAnnounce(`Training finished at step ${fmtInt.format(C.trainSteps)}. Greedy evaluation return ${fmtRet(trainerRef.current!.evaluate())}.`);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, speed, advance]);

  const togglePlay = () => {
    if (done) return;
    setPlaying((p) => {
      if (p) setAnnounce(`Paused at step ${fmtInt.format(trainer.trainStep)}.`);
      return !p;
    });
  };
  const stepOnce = () => {
    setPlaying(false);
    advance(1);
    bump();
  };
  const reset = () => {
    trainerRef.current = new GridWorldTrainer();
    historyRef.current = [{ step: 0, value: trainerRef.current.evaluate() }];
    firstOptimalRef.current = null;
    setPlaying(false);
    setAnnounce("Reset to step 0. Q-table cleared.");
    bump();
  };

  // --- derived view state ---------------------------------------------------
  const agent = trainer.agent;
  const rollout = trainer.greedyRollout();
  const evalNow = rollout.return;
  const hist = historyRef.current;
  // The greedy path, with repeated positions (bumping a wall) collapsed.
  const pathPoints: Obs[] = [];
  for (const p of rollout.path) {
    const prev = pathPoints[pathPoints.length - 1];
    if (!prev || prev.x !== p.x || prev.y !== p.y) pathPoints.push(p);
  }

  const selQ = [0, 1, 2, 3].map((a) => agent.qValue(selected, a));
  const selBest = agent.bestActionAndValue(selected)[0];
  const selIsGoal = selected.x === GOAL.x && selected.y === GOAL.y;
  const smoothAgents = !reduced && playing && speed === "1x";

  const onGridKey = (e: React.KeyboardEvent) => {
    const d: Record<string, [number, number]> = { ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] };
    const m = d[e.key];
    if (!m) return;
    e.preventDefault();
    setSelected((s) => ({ x: Math.min(SIZE - 1, Math.max(0, s.x + m[0])), y: Math.min(SIZE - 1, Math.max(0, s.y + m[1])) }));
  };

  const linePath = (() => {
    let d = `M${chartX(hist[0].step)} ${chartY(hist[0].value)}`;
    for (let i = 1; i < hist.length; i++) d += ` H${chartX(hist[i].step)} V${chartY(hist[i].value)}`;
    d += ` H${chartX(trainer.trainStep)}`;
    return d;
  })();
  const endX = chartX(trainer.trainStep);
  const endY = chartY(hist[hist.length - 1].value);

  return (
    <div ref={rootRef} className="card spot overflow-hidden bg-surface/90 shadow-[0_40px_120px_-40px_rgb(var(--ember)/0.25)]" data-offscreen={!inView}>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-5">
        <div className="flex items-center gap-2.5">
          <span className={`h-2 w-2 rounded-full ${running ? "bg-ember shadow-[0_0_12px_rgb(var(--ember))]" : done ? "bg-ok" : "bg-ink-3"}`} />
          <span className="font-mono text-xs text-ink-2">
            GridWorld 5×5 · tabular Q-learning · {C.numEnvs} lanes
          </span>
        </div>
        <span className="font-mono text-[0.7rem] text-ink-3">seed {C.agentSeed} · train seed {C.trainSeed}</span>
      </div>

      <div className="grid grid-cols-1 gap-0 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.12fr)]">
        {/* ---------------- grid ---------------- */}
        <div className="border-b border-line p-4 sm:p-5 lg:border-b-0 lg:border-r">
          <svg
            viewBox={`0 0 ${SIZE * CELL} ${SIZE * CELL}`}
            className="mx-auto block aspect-square w-full max-w-[26rem] touch-manipulation rounded-2xl"
            role="group"
            aria-label="GridWorld heatmap of V(s) = max Q(s, a) with greedy-policy arrows. Arrow keys move the inspected cell; its Q-values are listed below."
            tabIndex={0}
            onKeyDown={onGridKey}
          >
            <defs>
              <radialGradient id="gw-goal" cx="0.5" cy="0.5" r="0.6">
                <stop offset="0" stopColor="#ffe2a8" />
                <stop offset="0.5" stopColor="rgb(var(--ember-2))" />
                <stop offset="1" stopColor="rgb(var(--ember))" />
              </radialGradient>
              <filter id="gw-glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="6" />
              </filter>
            </defs>

            {Array.from({ length: SIZE * SIZE }, (_, i) => {
              const x = i % SIZE;
              const y = Math.floor(i / SIZE);
              const isGoal = x === GOAL.x && y === GOAL.y;
              const [best, v] = agent.bestActionAndValue({ x, y });
              const untouched = [0, 1, 2, 3].every((a) => agent.qValue({ x, y }, a) === 0);
              const hot = v > 6.2;
              const isSel = selected.x === x && selected.y === y;
              const [dx, dy] = DIRS[best];
              const X = cx(x);
              const Y = cy(y);
              return (
                <g
                  key={i}
                  aria-hidden
                  onPointerEnter={() => setSelected({ x, y })}
                  onClick={() => setSelected({ x, y })}
                  className="cursor-pointer"
                >
                  <rect
                    x={x * CELL + 3}
                    y={(SIZE - 1 - y) * CELL + 3}
                    width={CELL - 6}
                    height={CELL - 6}
                    rx={14}
                    fill={isGoal ? "url(#gw-goal)" : heat(v)}
                    stroke={isSel ? "rgb(var(--amber))" : "rgb(255 255 255 / 0.06)"}
                    strokeWidth={isSel ? 3 : 1}
                  />
                  {isGoal ? null : (
                    <>
                      <path
                        d={arrowPath(X, Y - 2, dx, dy)}
                        fill="none"
                        stroke={hot ? "#1a0b04" : "rgb(var(--ink))"}
                        strokeOpacity={untouched ? 0.28 : 0.9}
                        strokeWidth={5}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <text
                        x={x * CELL + 13}
                        y={(SIZE - 1 - y) * CELL + 26}
                        fontSize="15"
                        fill={hot ? "#1a0b04" : "rgb(var(--ink-2))"}
                        fillOpacity={untouched ? 0.55 : 0.95}
                        className="font-mono"
                      >
                        {fmtQ(v)}
                      </text>
                    </>
                  )}
                </g>
              );
            })}

            {/* greedy evaluation path from (0,0) */}
            <polyline
              points={pathPoints.map((p) => `${cx(p.x)},${cy(p.y)}`).join(" ")}
              fill="none"
              stroke={rollout.reachedGoal ? "rgb(var(--amber))" : "rgb(var(--steel))"}
              strokeWidth={4}
              strokeDasharray={rollout.reachedGoal ? undefined : "2 10"}
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity={0.85}
              pointerEvents="none"
            />
            <circle cx={cx(0)} cy={cy(0)} r={7} fill="rgb(var(--steel))" pointerEvents="none" />

            {/* goal marker, drawn above the path */}
            <g pointerEvents="none">
              <circle cx={cx(GOAL.x)} cy={cy(GOAL.y)} r={26} fill="#ffd28a" opacity={0.5} filter="url(#gw-glow)" className="ember-pulse" />
              <rect x={cx(GOAL.x) - 30} y={cy(GOAL.y) - 17} width={60} height={34} rx={17} fill="#ffe2a8" />
              <text x={cx(GOAL.x)} y={cy(GOAL.y) + 8} textAnchor="middle" fontSize="23" fontWeight="700" fill="#1a0b04" className="font-mono">
                +10
              </text>
            </g>

            {/* the 4 training lanes */}
            {trainer.observations.map((o, lane) => (
              <g
                key={lane}
                pointerEvents="none"
                style={{
                  transform: `translate(${cx(o.x) + LANE_OFFSETS[lane][0]}px, ${cy(o.y) + LANE_OFFSETS[lane][1]}px)`,
                  transition: smoothAgents ? "transform 170ms cubic-bezier(0.34,1.56,0.64,1)" : "none",
                }}
              >
                <circle r={9} fill="rgb(var(--bg))" stroke="rgb(var(--steel))" strokeWidth={2.5} />
                <text y={4} textAnchor="middle" fontSize="11" fontWeight="700" fill="rgb(var(--steel))" className="font-mono">
                  {lane}
                </text>
              </g>
            ))}
          </svg>

          {/* selected-cell readout */}
          <div className="mx-auto mt-3 max-w-[26rem] rounded-xl border border-line bg-bg-2/80 px-3 py-2.5" aria-live="off">
            <div className="flex items-center justify-between font-mono text-[0.72rem] text-ink-3">
              <span>
                Q(x={selected.x}, y={selected.y}, ·)
              </span>
              <span>{selIsGoal ? "terminal: never acted from" : `greedy → ${ACTION_NAMES[selBest]}`}</span>
            </div>
            <div className="mt-2 grid grid-cols-4 gap-2">
              {ACTION_NAMES.map((name, a) => (
                <div key={name} className="min-w-0">
                  <div className={`truncate font-mono text-[0.68rem] ${a === selBest && !selIsGoal ? "text-amber" : "text-ink-3"}`}>{name}</div>
                  <div className={`font-mono text-sm tabular-nums ${a === selBest && !selIsGoal ? "text-ink" : "text-ink-2"}`}>{fmtQ(selQ[a])}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ---------------- stats + chart + controls ---------------- */}
        <div className="flex min-w-0 flex-col p-4 sm:p-5">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            <Stat label="trainer step" value={`${fmtInt.format(trainer.trainStep)}`} sub={`/ ${fmtInt.format(C.trainSteps)}`} />
            <Stat label={<>explore <span className="normal-case">ε</span></>} value={agent.epsilon().toFixed(3)} sub={`${fmtInt.format(agent.stepCount)} transitions`} />
            <Stat label="episodes done" value={fmtInt.format(trainer.episodesCompleted)} sub="across 4 lanes" />
            <Stat
              label="greedy return"
              value={fmtRet(evalNow)}
              sub={evalNow === 3 ? "optimal" : evalNow === -100 ? "stuck at a wall" : "not yet optimal"}
              hot={evalNow === 3}
            />
          </dl>

          <figure ref={figRef} className="mt-5">
            <svg viewBox={`0 0 ${chartWidth} ${CH.h}`} className="block w-full" role="img" aria-label={`Evaluation return over training: starts at −100.0, currently ${fmtRet(evalNow)}.`}>
              {[0, -50, -100].map((v) => (
                <g key={v}>
                  <line x1={CH.l} x2={chartWidth - CH.r} y1={chartY(v)} y2={chartY(v)} stroke="rgb(var(--line))" strokeDasharray={v === 0 ? undefined : "3 5"} />
                  <text x={CH.l - 8} y={chartY(v) + 4} textAnchor="end" fontSize="11" fill="rgb(var(--ink-3))" className="font-mono">
                    {v < 0 ? `−${-v}` : v}
                  </text>
                </g>
              ))}
              <line x1={CH.l} x2={chartWidth - CH.r} y1={chartY(3)} y2={chartY(3)} stroke="rgb(var(--amber))" strokeOpacity={0.55} strokeDasharray="6 6" />
              <text x={chartWidth - CH.r} y={chartY(3) - 7} textAnchor="end" fontSize="11" fill="rgb(var(--amber))" className="font-mono">
                optimal 3.0
              </text>
              {X_TICKS.map(([s, label]) => (
                <text key={s} x={chartX(s)} y={CH.h - 26} textAnchor="middle" fontSize="11" fill="rgb(var(--ink-3))" className="font-mono">
                  {label}
                </text>
              ))}
              <text x={chartWidth - CH.r} y={CH.h - 4} textAnchor="end" fontSize="10.5" fill="rgb(var(--ink-3))" className="font-mono">
                trainer step, log scale →
              </text>
              <path d={linePath} fill="none" stroke="url(#gw-line)" strokeWidth={2.5} strokeLinejoin="round" />
              <defs>
                <linearGradient id="gw-line" x1="0" x2="1" y1="0" y2="0">
                  <stop offset="0" stopColor="rgb(var(--steel))" />
                  <stop offset="1" stopColor="rgb(var(--ember-2))" />
                </linearGradient>
              </defs>
              <circle cx={chartX(0)} cy={chartY(hist[0].value)} r={4} fill="rgb(var(--steel))" />
              <text x={chartX(0) + 8} y={chartY(hist[0].value) - 8} fontSize="11" fill="rgb(var(--steel))" className="font-mono">
                {fmtRet(hist[0].value)}
              </text>
              <circle cx={endX} cy={endY} r={4.5} fill={evalNow === 3 ? "rgb(var(--amber))" : "rgb(var(--ink-2))"} />
            </svg>
            <figcaption className="mt-1 flex flex-wrap justify-between gap-2 font-mono text-[0.7rem] text-ink-3">
              <span>mean greedy return, {C.evalEpisodes} eval episodes (seed {C.evalSeed}), after every trainer step</span>
              {firstOptimalRef.current !== null ? <span className="text-amber">first 3.0 at step {fmtInt.format(firstOptimalRef.current)}</span> : null}
            </figcaption>
          </figure>

          <div className="mt-5 flex flex-wrap items-center gap-2">
            <button type="button" className="ctl min-w-[5.8rem]" onClick={togglePlay} disabled={done} aria-pressed={playing}>
              {playing ? <PauseIcon className="h-3.5 w-3.5" /> : <PlayIcon className="h-3.5 w-3.5" />}
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" className="ctl" onClick={stepOnce} disabled={done}>
              <StepIcon className="h-3.5 w-3.5" /> Step once
            </button>
            <button type="button" className="ctl" onClick={reset}>
              <ResetIcon className="h-3.5 w-3.5" /> Reset
            </button>
            <div role="group" aria-label="Speed" className="ml-auto flex rounded-full border border-line-2 bg-bg-2 p-0.5">
              {SPEEDS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  aria-pressed={speed === s.id}
                  onClick={() => setSpeed(s.id)}
                  title={`${s.sps} trainer steps per second`}
                  className={`rounded-full px-2.5 py-1.5 font-mono text-[0.72rem] transition-colors ${
                    speed === s.id ? "bg-amber/15 text-amber" : "text-ink-3 hover:text-ink"
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {done ? (
            <p className="mt-4 rounded-xl border border-ok/30 bg-ok/5 px-3 py-2 text-sm text-ink-2">
              <span className="font-semibold text-ok">20,000 steps, greedy return {fmtRet(evalNow)}.</span> The C++ test asserts
              exactly this (<code className="font-mono text-[0.8rem]">tests/test_trainer_grid_world.cpp:96</code>).
            </p>
          ) : null}

          <p className="mt-auto pt-5 text-[0.8rem] leading-relaxed text-ink-3">
            <span className="font-semibold text-ink-2">A TypeScript port, not the C++ library running in your browser.</span> Same
            GridWorld, 4-lane SyncVectorEnvironment, TabularQLearningAgent and Trainer, same config as{" "}
            <code className="font-mono text-[0.75rem] text-ink-2">tests/test_trainer_grid_world.cpp</code>: lr 0.1, γ 0.99, ε 1.0→0.05
            over 20,000 transitions, slip 0, ties to Up. Checked checkpoint-for-checkpoint against a native run in{" "}
            <code className="font-mono text-[0.75rem] text-ink-2">web/scripts/parity</code>.
          </p>
        </div>
      </div>
      <p className="sr-only" aria-live="polite">
        {announce}
      </p>
    </div>
  );
}

function Stat({ label, value, sub, hot }: { label: React.ReactNode; value: string; sub: string; hot?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="font-mono text-[0.66rem] uppercase tracking-[0.14em] text-ink-3">{label}</dt>
      <dd className={`mt-0.5 font-mono text-xl tabular-nums sm:text-[1.35rem] ${hot ? "text-hot font-semibold" : "text-ink"}`}>{value}</dd>
      <dd className="truncate font-mono text-[0.68rem] text-ink-3">{sub}</dd>
    </div>
  );
}
