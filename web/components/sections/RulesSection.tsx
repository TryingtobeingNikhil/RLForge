import { BellmanToggle } from "../BellmanToggle";
import { VersionGuardDemo } from "../VersionGuardDemo";
import { Section, SectionHeader, Src } from "../Section";
import { REPO } from "@/lib/content";

export function RulesSection() {
  return (
    <Section id="rules" className="bg-bg-2/40">
      <SectionHeader id="rules" eyebrow="Two rules we refused to break" title="Correctness you can't opt out of." aside="Both are cheap to skip. Neither is optional here." />

      <div className="mt-14 grid grid-cols-1 gap-10 lg:grid-cols-2 lg:gap-8">
        <article data-reveal className="flex flex-col">
          <p className="font-mono text-sm text-amber">Rule 1</p>
          <h3 className="mt-2 text-2xl font-semibold tracking-[-0.01em] text-ink">
            <code className="font-mono text-[0.9em]">terminated</code> and <code className="font-mono text-[0.9em]">truncated</code> are never the same flag.
          </h3>
          {/* README.md:75-81 */}
          <p className="mt-3 text-[0.98rem] leading-relaxed text-ink-2">
            An agent that died and an episode that hit a time limit look identical if all you track is{" "}
            <code className="font-mono text-ink">done</code>. The Bellman equation disagrees: one zeroes the bootstrap, the other keeps it.
            Every environment, buffer and batch conversion in RLForge keeps them apart.
          </p>
          <div className="mt-6 flex-1">
            <BellmanToggle />
          </div>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
            <Src href={`${REPO.blob}/tests/test_tabular_q_learning_agent.cpp#L37-L112`}>tests/test_tabular_q_learning_agent.cpp:37-112</Src>
            <Src href={`${REPO.blob}/src/agents/tabular_q_learning_agent.cpp#L120-L123`}>src/agents/tabular_q_learning_agent.cpp:120</Src>
            <Src href={`${REPO.blob}/include/rl/data/batch_to_tensors.hpp#L42-L51`}>include/rl/data/batch_to_tensors.hpp:42 (DQN)</Src>
          </div>
        </article>

        <article data-reveal className="flex flex-col" style={{ ["--reveal-delay" as string]: "120ms" }}>
          <p className="font-mono text-sm text-amber">Rule 2</p>
          <h3 className="mt-2 text-2xl font-semibold tracking-[-0.01em] text-ink">Mutate a tensor after forward, and backward throws.</h3>
          {/* README.md:84-91 */}
          <p className="mt-3 text-[0.98rem] leading-relaxed text-ink-2">
            Every storage carries a version counter; every backward closure captures the version it saw at forward time. If they
            disagree, RLForge throws instead of handing you a gradient computed against data that no longer exists.
          </p>
          <div className="mt-6 flex-1">
            <VersionGuardDemo />
          </div>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
            <Src href={`${REPO.blob}/tests/test_version_guard.cpp#L27-L59`}>tests/test_version_guard.cpp:27</Src>
            <Src href={`${REPO.blob}/src/tensor/tensor.cpp#L67-L79`}>src/tensor/tensor.cpp:67 check_version</Src>
            <Src href={`${REPO.blob}/include/rl/tensor/tensor.hpp#L17-L46`}>include/rl/tensor/tensor.hpp:17 Storage</Src>
          </div>
        </article>
      </div>

      <p className="aside mx-auto mt-16 max-w-2xl text-center text-2xl leading-snug text-ink-2 sm:text-3xl" data-reveal>
        &ldquo;…skipping it means your bugs show up as &lsquo;training is unstable&rsquo; three weeks later instead of a stack trace today.&rdquo;
      </p>
      <div className="mt-3 text-center">
        <Src href={`${REPO.blob}/README.md#two-rules-we-refused-to-break`}>README.md, Two rules we refused to break</Src>
      </div>
    </Section>
  );
}
