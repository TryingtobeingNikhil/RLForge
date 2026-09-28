import { ArchitectureMap } from "../ArchitectureMap";
import { CodeBlock } from "../CodeBlock";
import { Section, SectionHeader, Src } from "../Section";
import { REPO } from "@/lib/content";

// Verbatim from docs/agent_trainer_api.md ("Trainer::train()'s loop, per step")
const LOOP = `actions = agent.act(current_observations, /*explore=*/true);
step_result = train_env.step(actions);
transitions = make_transitions(current_observations, actions, step_result); // Milestone 3's bridge
agent.observe_transitions(transitions);
if (agent.should_update()) {
    metrics = agent.update();
}
current_observations = step_result.observations;`;

export function ArchitectureSection() {
  return (
    <Section id="architecture" className="bg-bg-2/40">
      <SectionHeader id="architecture" eyebrow="Architecture" title="Two stacks, one loop." aside="Experience on one side, math on the other, algorithms where they meet.">
        {/* docs/replay_buffer_api.md "Design intent"; docs/agent_trainer_api.md "Agent" (should_update) */}
        <p>
          <code className="font-mono text-ink">ReplayBuffer</code>&apos;s source never names a concrete storage type, so a new layout is a new{" "}
          <code className="font-mono text-ink">TransitionStorage</code>, not an edit. <code className="font-mono text-ink">Trainer</code> never asks
          whether an algorithm is step-based or rollout-based; it just asks <code className="font-mono text-ink">should_update()</code>. Pick a box
          to see what it uses and what depends on it.
        </p>
      </SectionHeader>

      <div className="mt-12" data-reveal>
        <ArchitectureMap />
      </div>

      <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)] lg:items-start" data-reveal>
        <CodeBlock
          code={LOOP}
          title="Trainer::train(), one step"
          source={<Src href={`${REPO.blob}/docs/agent_trainer_api.md#trainer-rlcoretrainerhpp`}>docs/agent_trainer_api.md · src/core/trainer.cpp:22-45</Src>}
        />
        <ul className="space-y-4 text-[0.95rem] leading-relaxed text-ink-2">
          <li className="card p-4">
            <span className="font-semibold text-ink">Training always goes through VectorEnvironment</span>, even with one env. There is no
            single-env code path, so PPO and threaded rollouts landed without touching Trainer: its files last changed in Milestone 4.
            {/* git log -- src/core/trainer.cpp include/rl/core/trainer.hpp -> d742cd9 only */}
            <span className="mt-2 block">
              <Src href={`${REPO.commit}/d742cd9`}>git log -- src/core/trainer.cpp → d742cd9 (Milestone 4)</Src>
            </span>
          </li>
          <li className="card p-4">
            <span className="font-semibold text-ink">Evaluation uses its own Environment</span>, so it never disturbs an in-flight training
            episode and never mutates the agent.
          </li>
          <li className="card p-4">
            <span className="font-semibold text-ink">Factories, not instances.</span> Each env is constructed inside its own worker, which
            is why the threaded version kept the same constructor signature.
          </li>
          <li>
            <Src href={`${REPO.blob}/docs/vector_environment_api.md`}>docs/vector_environment_api.md · docs/threaded_vector_environment_api.md</Src>
          </li>
        </ul>
      </div>
    </Section>
  );
}
