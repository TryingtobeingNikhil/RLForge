"use client";

import { useMemo, useState } from "react";
import { REPO } from "@/lib/content";

// Derived from include/rl/**. "uses" edges follow #include relationships
// (e.g. include/rl/agents/dqn.hpp includes qnetwork.hpp, adam.hpp,
// replay_buffer.hpp; src/agents/dqn.cpp includes batch_to_tensors.hpp;
// src/core/trainer.cpp includes transition.hpp).
type Lane = "experience" | "loop" | "math";

interface ArchNode {
  id: string;
  name: string;
  lane: Lane;
  headers: string[];
  what: string;
  impls?: string[];
  uses: string[];
}

const NODES: ArchNode[] = [
  // experience
  { id: "env", name: "Environment", lane: "experience", headers: ["core/environment.hpp", "envs/grid_world.hpp"], what: "reset(seed) / step(action). Every step reports terminated and truncated separately.", impls: ["GridWorld"], uses: [] },
  { id: "vec", name: "VectorEnvironment", lane: "experience", headers: ["core/vector_environment.hpp", "vector_envs/"], what: "N sub-envs built from factories. Seeds seed + i, auto-resets, keeps final_observations.", impls: ["SyncVectorEnvironment", "ThreadedVectorEnvironment"], uses: ["env"] },
  { id: "transition", name: "Transition", lane: "experience", headers: ["core/transition.hpp"], what: "make_transitions() substitutes the true final observation at every episode boundary.", uses: ["vec"] },
  { id: "buffers", name: "ReplayBuffer · RolloutBuffer", lane: "experience", headers: ["core/replay_buffer.hpp", "replay_buffers/"], what: "A ring buffer over a pluggable TransitionStorage; a lane-preserving rollout buffer for GAE.", impls: ["VectorTransitionStorage"], uses: ["transition"] },
  // loop
  { id: "trainer", name: "Trainer", lane: "loop", headers: ["core/trainer.hpp"], what: "act → step → make_transitions → observe → update when should_update(). Evaluates on a separate env.", uses: ["vec", "env", "transition", "agent"] },
  { id: "agent", name: "Agent", lane: "loop", headers: ["core/agent.hpp"], what: "Four hooks: act, observe_transitions, should_update, update. Step-based and rollout-based algorithms share one loop.", uses: ["transition"] },
  { id: "tabular", name: "TabularQLearningAgent", lane: "loop", headers: ["agents/tabular_q_learning_agent.hpp"], what: "ε-greedy tabular Q-learning. The reference agent behind −100 → 3.0.", uses: ["agent"] },
  { id: "dqn", name: "DQNAgent", lane: "loop", headers: ["agents/dqn.hpp", "agents/epsilon_greedy.hpp"], what: "QNetwork + hard-synced target net, Adam, experience replay, (1 − terminated) bootstrap mask.", uses: ["agent", "buffers", "nn", "optim", "bridge"] },
  { id: "ppo", name: "PPOAgent", lane: "loop", headers: ["agents/ppo.hpp"], what: "Actor-critic, clipped surrogate, lane-correct GAE, shuffled minibatches over multiple epochs.", uses: ["agent", "buffers", "nn", "optim"] },
  // math
  { id: "backend", name: "TensorBackend", lane: "math", headers: ["tensor/backend.hpp"], what: "matmul dispatch: portable CPU by default, optional CBLAS and CUDA/cuBLAS.", impls: ["cpu", "blas", "cuda"], uses: [] },
  { id: "tensor", name: "Tensor + autograd", lane: "math", headers: ["tensor/tensor.hpp", "tensor/autograd.hpp"], what: "Contiguous double storage with a version counter; define-by-run reverse mode; no_grad().", uses: ["backend"] },
  { id: "nn", name: "nn", lane: "math", headers: ["nn/"], what: "Module, Linear (Kaiming init), QNetwork, ActorCriticNetwork, mse_loss.", uses: ["tensor"] },
  { id: "optim", name: "optim", lane: "math", headers: ["optim/"], what: "Optimizer base, SGD with momentum, Adam with bias correction.", uses: ["tensor"] },
  { id: "bridge", name: "batch_to_tensors", lane: "math", headers: ["data/batch_to_tensors.hpp"], what: "Sampled TransitionBatch → Tensors, with a terminated mask that ignores truncation.", uses: ["tensor", "transition"] },
];

const LANES: { id: Lane; title: string; sub: string }[] = [
  { id: "experience", title: "Experience", sub: "rl::core · envs · vector_envs · replay_buffers" },
  { id: "loop", title: "The loop", sub: "rl::core::Trainer · rl::agents" },
  { id: "math", title: "Math", sub: "rl::tensor · nn · optim · data" },
];

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

export function ArchitectureMap() {
  const [active, setActive] = useState<string>("trainer");
  const node = byId[active];

  const related = useMemo(() => {
    const uses = new Set(node.uses);
    const usedBy = new Set(NODES.filter((n) => n.uses.includes(active)).map((n) => n.id));
    return { uses, usedBy };
  }, [active, node]);

  return (
    <div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {LANES.map((lane) => (
          <div key={lane.id} className={`rounded-[1.6rem] border border-line p-3 ${lane.id === "loop" ? "bg-ember/[0.035]" : "bg-surface/40"}`}>
            <div className="px-2 pb-3 pt-1">
              <p className={`font-semibold ${lane.id === "loop" ? "text-amber" : lane.id === "math" ? "text-steel" : "text-ink"}`}>{lane.title}</p>
              <p className="font-mono text-[0.68rem] text-ink-3">{lane.sub}</p>
            </div>
            <ul className="space-y-2">
              {NODES.filter((n) => n.lane === lane.id).map((n) => {
                const isActive = n.id === active;
                const isUse = related.uses.has(n.id);
                const isUsedBy = related.usedBy.has(n.id);
                const dim = !isActive && !isUse && !isUsedBy;
                return (
                  <li key={n.id}>
                    <button
                      type="button"
                      onClick={() => setActive(n.id)}
                      onMouseEnter={() => setActive(n.id)}
                      onFocus={() => setActive(n.id)}
                      aria-pressed={isActive}
                      className={`group relative w-full rounded-2xl border px-3.5 py-3 text-left transition-all duration-300 ease-out ${
                        isActive
                          ? "border-amber/70 bg-surface-2 shadow-[0_10px_40px_-16px_rgb(var(--ember)/0.6)]"
                          : isUse
                            ? "border-steel/50 bg-surface-2/80"
                            : isUsedBy
                              ? "border-ember-2/40 bg-surface-2/80"
                              : "border-line bg-surface/70"
                      } ${dim ? "opacity-55" : "opacity-100"}`}
                    >
                      <span className="flex items-center justify-between gap-2">
                        <span className="truncate font-mono text-[0.86rem] font-semibold text-ink">{n.name}</span>
                        {isUse ? <span className="shrink-0 font-mono text-[0.64rem] text-steel">dependency</span> : null}
                        {isUsedBy ? <span className="shrink-0 font-mono text-[0.64rem] text-ember-2">depends on it</span> : null}
                      </span>
                      <span className="mt-0.5 block truncate font-mono text-[0.68rem] text-ink-3">rl/{n.headers[0]}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      <div className="card spot mt-4 grid grid-cols-1 gap-4 p-5 sm:p-6 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]" aria-live="polite">
        <div className="min-w-0">
          <p className="font-mono text-lg font-semibold text-amber">{node.name}</p>
          <p className="mt-2 text-[0.98rem] leading-relaxed text-ink-2">{node.what}</p>
          {node.impls ? (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {node.impls.map((i) => (
                <span key={i} className="chip">
                  {i}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="min-w-0 space-y-3 font-mono text-[0.75rem]">
          <div>
            <p className="text-ink-3">headers</p>
            <ul className="mt-1 space-y-0.5">
              {node.headers.map((h) => (
                <li key={h} className="truncate">
                  <a className="link text-ink-2" href={`${h.endsWith("/") ? REPO.tree : REPO.blob}/include/rl/${h}`}>
                    include/rl/{h}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-steel">uses</p>
              <p className="mt-1 text-ink-2">{node.uses.length ? node.uses.map((u) => byId[u].name.split(" ")[0]).join(", ") : "nothing above it"}</p>
            </div>
            <div>
              <p className="text-ember-2">used by</p>
              <p className="mt-1 text-ink-2">
                {related.usedBy.size ? [...related.usedBy].map((u) => byId[u].name.split(" ")[0]).join(", ") : "the caller"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
