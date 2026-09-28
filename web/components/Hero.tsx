import { ArrowIcon, GitHubIcon } from "./Icons";
import { GridWorldDemo } from "./GridWorldDemo";
import { StarCount } from "./RepoStatsProvider";
import { REPO, TAGLINE, TOTAL_TESTS } from "@/lib/content";

const BADGES = [
  { label: "C++20", src: "CMakeLists.txt:6 (CMAKE_CXX_STANDARD 20)" },
  { label: `${TOTAL_TESTS} tests`, src: "counted from tests/*.cpp; README.md:12 badge" },
  { label: "zero dependencies", src: "README.md:13 badge; Catch2 is fetched for tests only (tests/CMakeLists.txt:2-7)" },
  { label: "-Werror", src: "CMakeLists.txt:17 (-Wall -Wextra -Wpedantic -Werror)" },
];

export function Hero() {
  return (
    <section id="top" className="relative pb-20 pt-10 sm:pt-16">
      <div className="forge-glow" aria-hidden />
      <div className="container-site relative">
        <p className="eyebrow" data-reveal>
          RLForge · reinforcement learning in C++20, from the tensor up
        </p>
        {/* README.md:7 */}
        <h1
          className="mt-5 max-w-5xl text-balance text-[2.6rem] font-semibold leading-[0.98] tracking-[-0.035em] text-ink sm:text-6xl lg:text-[5.2rem]"
          data-reveal
          style={{ ["--reveal-delay" as string]: "60ms" }}
        >
          We built the tensor engine so we didn&apos;t have to trust <span className="text-hot">anyone else&apos;s gradients.</span>
          <span className="sr-only"> ({TAGLINE})</span>
        </h1>
        <p className="lede mt-6" data-reveal style={{ ["--reveal-delay" as string]: "120ms" }}>
          {/* README.md "Read this before you judge the checkmarks" + milestone list */}
          A from-scratch reinforcement-learning library: its own tensor and reverse-mode autograd, NN layers and
          optimizers, tabular Q-learning, DQN and PPO, sync and threaded vector environments, and optional CBLAS and
          CUDA backends. No <code className="font-mono text-[0.92em] text-ink">import torch</code>. No Eigen. No borrowed
          autograd.
        </p>

        <ul className="mt-7 flex flex-wrap gap-2" data-reveal style={{ ["--reveal-delay" as string]: "180ms" }}>
          {BADGES.map((b) => (
            <li key={b.label} className="chip" title={`Source: ${b.src}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-ember" aria-hidden />
              {b.label}
            </li>
          ))}
        </ul>

        <div className="mt-8 flex flex-wrap items-center gap-3" data-reveal style={{ ["--reveal-delay" as string]: "240ms" }}>
          <a href={REPO.url} className="btn-hot">
            <GitHubIcon className="h-4 w-4" />
            Star on GitHub
            <StarCount className="inline-flex items-center gap-1 rounded-full bg-black/15 px-2 py-0.5 font-mono text-xs" />
          </a>
          <a href="#quickstart" className="btn-ghost group">
            Quickstart
            <ArrowIcon className="h-4 w-4 transition-transform duration-300 ease-spring group-hover:translate-x-1" />
          </a>
        </div>

        <div className="mt-14" data-reveal style={{ ["--reveal-delay" as string]: "120ms" }}>
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <h2 className="max-w-xl text-xl font-semibold tracking-[-0.01em] text-ink sm:text-2xl">
              Watch an agent learn GridWorld,{" "}
              <span className="aside font-normal text-ink-2">from −100 to optimal.</span>
            </h2>
            <p className="max-w-sm text-sm text-ink-3">
              Start (0,0), goal (4,4). −1 per step, +10 at the goal, 100-step limit. Hover or tap a cell for its Q-values.
            </p>
          </div>
          <GridWorldDemo />
        </div>
      </div>
    </section>
  );
}
