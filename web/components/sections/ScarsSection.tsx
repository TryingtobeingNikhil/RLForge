import { Section, SectionHeader, Src } from "../Section";
import { NOT_BUILT, REPO } from "@/lib/content";

// An illustration of the failure mode, not the original file: a comment
// ending in a backslash splices the next source line into the comment.
const SPLICE = [
  { t: "//    x", c: "" },
  { t: "//   / \\", c: "← the line ends in a backslash…" },
  { t: "auto a = x.mul(2.0);", c: "…so this line is now part of the comment", swallowed: true },
];

export function ScarsSection() {
  return (
    <Section id="scars">
      <div className="grid grid-cols-1 gap-12 lg:grid-cols-2 lg:gap-10">
        <div>
          <SectionHeader id="scars" eyebrow="Scars" title="We're not going to pretend this shipped clean." aside="Honesty is a feature. So is -Werror.">
            {/* README.md:159-172 */}
            <p>
              <code className="font-mono text-ink">-Wall -Wextra -Wpedantic -Werror</code> has been on since commit one. At one point a stray
              backslash at the end of a comment in a test file, meant as harmless ASCII art, got read by the compiler as a line continuation
              and broke the build on a fresh clone. <span className="text-ink">-Werror caught it immediately.</span>
            </p>
            <p>
              The bug that costs you five minutes today is the one that would&apos;ve cost someone else an afternoon of &ldquo;why won&apos;t
              this compile&rdquo; next month.
            </p>
          </SectionHeader>

          <figure className="mt-8" data-reveal>
            <div className="code !p-0">
              {SPLICE.map((l, i) => (
                <div key={i} className={`flex items-baseline gap-4 whitespace-nowrap px-4 py-1 ${i === 0 ? "pt-3" : ""} ${i === SPLICE.length - 1 ? "pb-3" : ""}`}>
                  <span className={l.swallowed ? "text-ink-3 line-through decoration-bad/70" : "italic text-ink-3"}>{l.t}</span>
                  {l.c ? <span className="font-sans text-[0.72rem] not-italic text-amber">{l.c}</span> : null}
                </div>
              ))}
            </div>
            <figcaption className="mt-2 text-[0.75rem] text-ink-3">
              An illustration of the failure mode, not the original file. GCC reports it under -Wcomment (on with -Wall);
              -Werror turns it into a stop.
            </figcaption>
          </figure>
          <div className="mt-3">
            <Src href={`${REPO.blob}/README.md#scars`}>README.md, Scars · CMakeLists.txt:17</Src>
          </div>
        </div>

        <div data-reveal style={{ ["--reveal-delay" as string]: "120ms" }}>
          <div className="card relative overflow-hidden p-6 sm:p-8">
            <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-ember/10 blur-3xl" aria-hidden />
            <p className="eyebrow">What we didn&apos;t build (yet)</p>
            <h3 className="mt-3 text-2xl font-semibold tracking-[-0.01em] text-ink sm:text-3xl">Documented, not hidden.</h3>
            <p className="mt-3 text-[0.95rem] leading-relaxed text-ink-2">
              Every deferred piece is written down: what it is, why it isn&apos;t here, and where it belongs.
            </p>
            <ul className="mt-6 space-y-2.5">
              {NOT_BUILT.map((n) => (
                <li key={n.item} className="flex items-start gap-3 rounded-2xl border border-dashed border-line-2 bg-bg-2/60 px-4 py-3">
                  <span className="mt-1.5 h-2 w-2 shrink-0 rounded-sm border border-ink-3" aria-hidden />
                  <span className="min-w-0">
                    <span className="block text-[0.95rem] text-ink">{n.item}</span>
                    <a href={`${REPO.blob}/${n.source}`} className="src hover:text-ink-2">
                      {n.source}
                    </a>
                  </span>
                </li>
              ))}
            </ul>
            {/* README.md:237-239 */}
            <p className="aside mt-6 text-xl leading-snug text-ink">
              If a homemade library&apos;s README doesn&apos;t have a section like this, it either doesn&apos;t have limitations or isn&apos;t
              telling you about them.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
