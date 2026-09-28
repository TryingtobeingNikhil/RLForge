import { Section, SectionHeader, Src } from "../Section";
import { GRIDWORLD_CONFIG as C, GridWorldTrainer, type Obs } from "@/lib/gridworld";
import { REPO } from "@/lib/content";

// Both paths are computed at build time by the same port the hero runs
// (lib/gridworld.ts), then drawn statically.
function computePaths() {
  const t = new GridWorldTrainer();
  const before = t.greedyRollout();
  while (!t.done) t.step();
  const after = t.greedyRollout();
  return { before, after, episodes: t.episodesCompleted };
}

const S = C.size;
const cell = 44;
const px = (x: number) => x * cell + cell / 2;
const py = (y: number) => (S - 1 - y) * cell + cell / 2;

function MiniGrid({ path, reached, label }: { path: Obs[]; reached: boolean; label: string }) {
  const pts: Obs[] = [];
  for (const p of path) {
    const prev = pts[pts.length - 1];
    if (!prev || prev.x !== p.x || prev.y !== p.y) pts.push(p);
  }
  return (
    <svg viewBox={`0 0 ${S * cell} ${S * cell}`} className="block aspect-square w-40 shrink-0 sm:w-44" role="img" aria-label={label}>
      {Array.from({ length: S * S }, (_, i) => {
        const x = i % S;
        const y = Math.floor(i / S);
        const goal = x === S - 1 && y === S - 1;
        return (
          <rect
            key={i}
            x={x * cell + 2}
            y={(S - 1 - y) * cell + 2}
            width={cell - 4}
            height={cell - 4}
            rx={8}
            fill={goal ? "rgb(var(--ember) / 0.9)" : "rgb(var(--surface-2))"}
            stroke="rgb(var(--line))"
          />
        );
      })}
      <polyline
        points={pts.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
        fill="none"
        stroke={reached ? "rgb(var(--amber))" : "rgb(var(--steel))"}
        strokeWidth={4}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {pts.slice(1).map((p, i) => (
        <circle key={i} cx={px(p.x)} cy={py(p.y)} r={3.5} fill={reached ? "rgb(var(--amber))" : "rgb(var(--steel))"} />
      ))}
      <circle cx={px(0)} cy={py(0)} r={7} fill="rgb(var(--ink))" />
      {!reached ? (
        <g>
          <circle cx={px(0)} cy={py(S - 1)} r={12} fill="none" stroke="rgb(var(--bad))" strokeWidth={2.5} strokeDasharray="3 4" />
        </g>
      ) : null}
    </svg>
  );
}

export function ProofSection() {
  const { before, after } = computePaths();
  const bumps = before.path.length - 1 - (S - 1); // steps that don't move

  return (
    <Section id="proof">
      <SectionHeader
        id="proof"
        eyebrow="The proof"
        title={
          <>
            <span className="text-steel">−100</span> <span className="text-ink-3">→</span> <span className="text-hot">3.0</span>, checked
            arithmetically.
          </>
        }
        aside="Not “the loss went down.” A proven-optimal policy."
      >
        {/* README.md:118-133, docs/agent_trainer_api.md "What this milestone proved" */}
        <p>
          <code className="font-mono text-ink">tests/test_trainer_grid_world.cpp</code> trains a tabular agent on a 4-way vectorized 5×5
          GridWorld for 20,000 steps with every seed fixed, and asserts two exact numbers.
        </p>
      </SectionHeader>

      <div className="mt-12 grid grid-cols-1 gap-5 md:grid-cols-2">
        <article className="card spot p-5 sm:p-7" data-reveal>
          <p className="eyebrow">Before training</p>
          <p className="mt-3 font-mono text-6xl font-semibold tracking-tight text-steel sm:text-7xl">−100.0</p>
          <div className="mt-6 flex flex-col gap-5 sm:flex-row sm:items-center">
            <MiniGrid path={before.path} reached={before.reachedGoal} label="Untrained greedy path: straight up to the wall, then stuck." />
            <p className="text-[0.95rem] leading-relaxed text-ink-2">
              Empty Q-table, every action reads 0.0, ties go to action 0: <span className="text-ink">Up</span>. Four moves up, then{" "}
              {bumps} bumps into the wall until the {C.maxEpisodeSteps}-step limit truncates the episode. 100 × (−1), every episode.
            </p>
          </div>
          <div className="mt-5">
            <Src href={`${REPO.blob}/tests/test_trainer_grid_world.cpp#L75-L76`}>tests/test_trainer_grid_world.cpp:76 · REQUIRE(mean(...) == Approx(-100.0f))</Src>
          </div>
        </article>

        <article className="card spot p-5 sm:p-7" data-reveal style={{ ["--reveal-delay" as string]: "120ms" }}>
          <p className="eyebrow">After 20,000 steps</p>
          <p className="mt-3 font-mono text-6xl font-semibold tracking-tight sm:text-7xl">
            <span className="text-hot">3.0</span>
          </p>
          <div className="mt-6 flex flex-col gap-5 sm:flex-row sm:items-center">
            <MiniGrid path={after.path} reached={after.reachedGoal} label="Trained greedy path: eight moves to the goal." />
            <p className="text-[0.95rem] leading-relaxed text-ink-2">
              The shortest path from (0,0) to (4,4) is 8 moves, the Manhattan distance. With no slip the greedy policy takes one every
              episode. This is the path the port&apos;s trained agent takes; any 8-move path scores the same.
            </p>
          </div>
          <div className="mt-5">
            <Src href={`${REPO.blob}/tests/test_trainer_grid_world.cpp#L89-L96`}>tests/test_trainer_grid_world.cpp:96 · REQUIRE(mean(...) == Approx(3.0f))</Src>
          </div>
        </article>
      </div>

      {/* 7 x (-1) + 10 = 3 — tests/test_trainer_grid_world.cpp:91-95, README.md:126-128 */}
      <div className="card mt-5 overflow-hidden p-5 sm:p-7" data-reveal>
        <p className="eyebrow">The arithmetic</p>
        <ol className="mt-5 grid grid-cols-4 gap-2 sm:grid-cols-8" aria-label="Rewards along the optimal path">
          {Array.from({ length: 8 }, (_, i) => (
            <li
              key={i}
              className={`flex flex-col items-center rounded-2xl border px-2 py-3 ${
                i === 7 ? "border-amber/60 bg-amber/10" : "border-line bg-surface-2"
              }`}
            >
              <span className="font-mono text-[0.66rem] text-ink-3">move {i + 1}</span>
              <span className={`mt-1 font-mono text-xl font-semibold ${i === 7 ? "text-amber" : "text-ink-2"}`}>{i === 7 ? "+10" : "−1"}</span>
            </li>
          ))}
        </ol>
        <p className="mt-6 whitespace-nowrap text-center font-mono text-2xl text-ink sm:text-5xl">
          7 × (−1) + 10 = <span className="text-hot font-semibold">3</span>
        </p>
        <p className="mx-auto mt-4 max-w-2xl text-center text-sm leading-relaxed text-ink-3">
          Reward is −1 per step and +10 on the step that reaches the goal (src/envs/grid_world.cpp:90). Deterministic dynamics mean a
          converged greedy policy hits this exactly on all 10 evaluation episodes, so the mean is exactly 3.0.
        </p>
      </div>
    </Section>
  );
}
