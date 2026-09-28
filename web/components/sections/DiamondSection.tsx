import { DiamondBug } from "../DiamondBug";
import { CodeBlock } from "../CodeBlock";
import { Section, SectionHeader, Src } from "../Section";
import { REPO } from "@/lib/content";

// Verbatim from tests/test_tensor.cpp:622-628
const TEST_EXCERPT = `auto x = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {4});
x.requires_grad_(true);

auto a = x.mul(2.0);   // a = 2x
auto b = x.mul(3.0);   // b = 3x
auto c = a.add(b);     // c = 5x  — x used via two paths
auto loss = c.mean();  // scalar`;

export function DiamondSection() {
  return (
    <Section id="diamond">
      <SectionHeader
        id="diamond"
        eyebrow="The diamond bug"
        title={<>The bug hiding in most &ldquo;from scratch&rdquo; autograd engines.</>}
        aside="Not a crash. Not an exception. Just a quietly wrong number."
      >
        {/* README.md:40-71 */}
        <p>
          <code className="font-mono text-ink">x</code> feeds two branches that meet again at{" "}
          <code className="font-mono text-ink">c</code>. The usual mistake: process a&apos;s branch, write x&apos;s
          gradient, move on, then <em>overwrite</em> it when b&apos;s branch reaches x a moment later. It works on every
          straight-line example and breaks the instant your graph isn&apos;t one.
        </p>
        <p>
          RLForge&apos;s rule: a node isn&apos;t processed until every consumer has deposited its contribution. Flip the
          toggle and watch the wrong gradient appear.
        </p>
      </SectionHeader>

      <div className="mt-12" data-reveal>
        <DiamondBug />
      </div>

      <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]" data-reveal>
        <CodeBlock
          code={TEST_EXCERPT}
          title="the exact graph, from the C++ test suite"
          source={
            <Src href={`${REPO.blob}/tests/test_tensor.cpp#L605-L637`}>
              tests/test_tensor.cpp:635 · REQUIRE((*x.grad())[i] == Approx(1.25))
            </Src>
          }
        />
        <div className="card spot p-5 sm:p-6">
          <p className="eyebrow">How RLForge guarantees it</p>
          <ol className="mt-4 space-y-3 text-[0.95rem] leading-relaxed text-ink-2">
            <li>
              <span className="font-mono text-amber">1.</span> Topological order by iterative post-order DFS. No
              recursion, so deep graphs can&apos;t blow the stack.
            </li>
            <li>
              <span className="font-mono text-amber">2.</span> Every node&apos;s{" "}
              <code className="font-mono text-ink">incoming_grad</code> is cleared, then contributions are{" "}
              <code className="font-mono text-ink">+=</code>&apos;d into it by{" "}
              <code className="font-mono text-ink">distribute_grad</code>.
            </li>
            <li>
              <span className="font-mono text-amber">3.</span> A node&apos;s{" "}
              <code className="font-mono text-ink">backward_fn</code> runs once, after all its consumers, under{" "}
              <code className="font-mono text-ink">no_grad()</code>.
            </li>
          </ol>
          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1">
            <Src href={`${REPO.blob}/src/tensor/tensor.cpp#L119-L158`}>src/tensor/tensor.cpp:119-158</Src>
            <Src href={`${REPO.blob}/docs/tensor_autograd_api.md`}>docs/tensor_autograd_api.md</Src>
          </div>
        </div>
      </div>
    </Section>
  );
}
