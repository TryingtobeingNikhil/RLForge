import { Section, SectionHeader, Src } from "../Section";
import { MILESTONES, PEOPLE, REPO, type Milestone } from "@/lib/content";

function renderHardPart(text: string) {
  // Render `code` spans from the README table.
  return text.split(/(`[^`]+`)/g).map((part, i) =>
    part.startsWith("`") ? (
      <code key={i} className="font-mono text-[0.92em] text-ink">
        {part.slice(1, -1)}
      </code>
    ) : (
      part
    ),
  );
}

function Credit({ who }: { who: Milestone["builtBy"] }) {
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      <a href={PEOPLE.nikhil.url} className="chip hover:border-ink-3/60">
        <span className="h-1.5 w-1.5 rounded-full bg-steel" aria-hidden />
        {PEOPLE.nikhil.name}
      </a>
      {who === "both" ? (
        <a href={PEOPLE.aprv10.url} className="chip border-amber/40 text-amber hover:border-amber/70">
          <span className="h-1.5 w-1.5 rounded-full bg-amber" aria-hidden />
          {PEOPLE.aprv10.name}
        </a>
      ) : null}
    </span>
  );
}

function MilestoneCard({ m }: { m: Milestone }) {
  return (
    <li className="relative pl-12 sm:pl-16" data-reveal>
      <span
        className={`absolute left-0 top-5 grid h-9 w-9 place-items-center rounded-full border font-mono text-sm font-semibold sm:h-11 sm:w-11 sm:text-base ${
          m.builtBy === "both" ? "border-amber/60 bg-[#231509] text-amber" : "border-line-2 bg-surface-2 text-ink"
        }`}
      >
        {m.n}
      </span>
      <article className="card spot p-5 sm:p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <h3 className="text-lg font-semibold tracking-[-0.01em] text-ink sm:text-xl">{m.title}</h3>
          <Credit who={m.builtBy} />
        </div>
        <p className="mt-3 text-[0.72rem] font-mono uppercase tracking-[0.14em] text-ink-3">The actual hard part</p>
        <p className="aside mt-1 text-xl leading-snug text-ink sm:text-[1.4rem]">{renderHardPart(m.hardPart)}</p>
        <ul className="mt-4 flex flex-wrap gap-1.5">
          {m.shipped.map((s) => (
            <li key={s} className="rounded-full bg-bg-2 px-2.5 py-1 font-mono text-[0.7rem] text-ink-2">
              {s}
            </li>
          ))}
        </ul>
        <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1">
          {m.doc ? <Src href={`${REPO.blob}/${m.doc}`}>{m.doc}</Src> : null}
          <Src href={`${REPO.commit}/${m.commit.sha}`}>
            {m.commit.sha} · {m.commit.label}
          </Src>
        </div>
      </article>
    </li>
  );
}

function Rail({ hot }: { hot?: boolean }) {
  return (
    <span
      className={`absolute bottom-8 left-[1.1rem] top-8 w-px sm:left-[1.37rem] ${hot ? "bg-amber/40" : "bg-gradient-to-b from-line-2 to-line-2/40"}`}
      aria-hidden
    />
  );
}

export function BuildLogSection() {
  const solo = MILESTONES.filter((m) => m.builtBy === "nikhil");
  const together = MILESTONES.filter((m) => m.builtBy === "both");
  return (
    <Section id="build-log">
      <div className="lg:grid lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.25fr)] lg:gap-14">
      <div className="lg:sticky lg:top-24 lg:self-start">
      <SectionHeader id="build-log" eyebrow="The build log" title="Ten milestones. Each one had a hard part." aside="Built incrementally, with every deferral written down.">
        {/* README.md "The build log" and its credits paragraph (README.md:151-155) */}
        <p>
          Milestones 1 through 7 (the environment stack, the tensor and autograd engine, and DQN) were built end-to-end by{" "}
          <a className="link" href={PEOPLE.nikhil.url}>
            {PEOPLE.nikhil.name}
          </a>
          . Milestones 8 through 10 (PPO, the threading layer, and the CUDA/BLAS backends) were built together with{" "}
          <a className="link" href={PEOPLE.aprv10.url}>
            {PEOPLE.aprv10.name}
          </a>
          , who led the concurrency and GPU work.
        </p>
      </SectionHeader>
      </div>

      <div className="relative mt-14 lg:mt-0">
        <div className="relative">
          <Rail />
          <ol className="space-y-5">
            {solo.map((m) => (
              <MilestoneCard key={m.n} m={m} />
            ))}
          </ol>
        </div>
        <div className="relative mt-10 rounded-[1.8rem] border border-amber/25 bg-amber/[0.03] p-3 pt-5 sm:p-4 sm:pt-6">
          <p className="mb-4 pl-12 text-sm text-ink-2 sm:pl-16">
            <span className="font-semibold text-amber">Built together with {PEOPLE.aprv10.name}</span>, who led concurrency and GPU work.
          </p>
          <div className="relative">
            <Rail hot />
            <ol className="space-y-5">
              {together.map((m) => (
                <MilestoneCard key={m.n} m={m} />
              ))}
            </ol>
          </div>
        </div>
      </div>
      </div>
    </Section>
  );
}
