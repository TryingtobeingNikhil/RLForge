import { CountUp } from "../CountUp";
import { NumGradDemo } from "../NumGradDemo";
import { Section, SectionHeader, Src } from "../Section";
import { TestBreakdown } from "../TestBreakdown";
import { REPO, TOTAL_TESTS } from "@/lib/content";

const TRIPLE = [
  {
    k: "Forward",
    t: "The value is right.",
    d: "Hand-computed expected outputs: square([−3, 0, 2, 4]) must come out as [9, 0, 4, 16].",
    src: "tests/test_tensor.cpp:221",
  },
  {
    k: "Analytic backward",
    t: "The derivative is right.",
    d: "The closed-form gradient each backward_fn must produce: d(x²)/dx at [1, 2, 3] is [2, 4, 6].",
    src: "tests/test_tensor.cpp:445",
  },
  {
    k: "Numerical gradient",
    t: "The math checks out.",
    d: "Central difference with ε = 1e-5 against backward(), relative tolerance 1e-5.",
    src: "tests/test_tensor.cpp:717",
  },
];

export function TestsSection() {
  return (
    <Section id="tests" className="bg-bg-2/40">
      <div className="grid grid-cols-1 gap-10 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
        <SectionHeader id="tests" eyebrow="The test suite" title="One binary. Tagged by subsystem." aside="“The math looks right” and “the math checks out numerically” are different claims.">
          {/* README.md:199-210 */}
          <p>Don&apos;t take our word for any of this. Clone it, build it, run ctest yourself.</p>
        </SectionHeader>
        <div className="text-left lg:text-right" data-reveal>
          <CountUp to={TOTAL_TESTS} className="block font-mono text-[6rem] font-semibold leading-none tracking-tight text-hot sm:text-[8rem]" />
          <p className="mt-2 font-mono text-xs text-ink-3">TEST_CASEs in rl_tests, counted from tests/*.cpp</p>
        </div>
      </div>

      <div className="mt-12" data-reveal>
        <TestBreakdown />
        <div className="mt-2">
          <Src href={`${REPO.tree}/tests`}>tests/ · web/tests/test_count.test.ts re-counts these from the C++ sources</Src>
        </div>
      </div>

      <div className="mt-16" data-reveal>
        <h3 className="text-2xl font-semibold tracking-[-0.01em] text-ink">The triple every core op has to pass.</h3>
        <p className="mt-3 max-w-3xl text-[0.98rem] leading-relaxed text-ink-2">
          Each Milestone 5 op (add, sub, elementwise and scalar mul, matmul, relu, square, mean, gather) has all three. Linear, mse_loss,
          transpose, broadcasting add and max_last_dim get numerical gradient checks too.
        </p>
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          {TRIPLE.map((c, i) => (
            <div key={c.k} className="card spot p-5">
              <p className="font-mono text-xs text-amber">0{i + 1}</p>
              <p className="mt-2 font-semibold text-ink">{c.k}</p>
              <p className="aside mt-1 text-lg">{c.t}</p>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{c.d}</p>
              <div className="mt-3">
                <Src href={`${REPO.blob}/${c.src.replace(/:(\d+)$/, "#L$1")}`}>{c.src}</Src>
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4">
          <NumGradDemo />
        </div>
      </div>
    </Section>
  );
}
