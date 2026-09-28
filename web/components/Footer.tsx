import { LogoMark, Wordmark } from "./Logo";
import { ForkCount, LicenseLine, StarCount } from "./RepoStatsProvider";
import { PEOPLE, REPO, TAGLINE } from "@/lib/content";

export function Footer() {
  return (
    <footer className="relative border-t border-line bg-bg-2/60">
      <div className="container-site grid grid-cols-1 gap-10 py-14 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <a href="#top" className="flex items-center gap-2.5 text-xl" aria-label="Back to top">
            <LogoMark className="h-9 w-9" animated={false} />
            <Wordmark />
          </a>
          <p className="aside mt-4 max-w-sm text-lg leading-snug text-ink-2">{TAGLINE}</p>
          <p className="mt-4 flex flex-wrap items-center gap-3 font-mono text-xs text-ink-3">
            <StarCount className="inline-flex items-center gap-1 text-amber" />
            <ForkCount />
          </p>
        </div>

        <div>
          <p className="eyebrow">Credits</p>
          {/* README.md:242-250 */}
          <p className="mt-3 text-sm leading-relaxed text-ink-2">
            Designed and primarily built by{" "}
            <a className="link" href={PEOPLE.nikhil.url}>
              {PEOPLE.nikhil.name}
            </a>{" "}
            (@{PEOPLE.nikhil.handle}): architecture, the environment, replay buffer and trainer stack, the tensor and autograd engine, NN
            layers and optimizers, and DQN.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-ink-2">
            PPO, the multi-threaded rollout system, and the CUDA/BLAS backend were built together with{" "}
            <a className="link" href={PEOPLE.aprv10.url}>
              {PEOPLE.aprv10.name}
            </a>
            , who led the concurrency and GPU work.
          </p>
        </div>

        <div>
          <p className="eyebrow">Links</p>
          <ul className="mt-3 space-y-2 text-sm">
            <li>
              <a className="link" href={REPO.url}>
                GitHub repository
              </a>
            </li>
            <li>
              <a className="link" href={`${REPO.tree}/docs`}>
                API docs
              </a>
            </li>
            <li>
              <a className="link" href={`${REPO.url}/issues`}>
                Issues
              </a>
            </li>
            <li>
              <a className="link" href={`${REPO.tree}/web`}>
                This site&apos;s source
              </a>
            </li>
          </ul>
          <p className="eyebrow mt-6">License</p>
          <p className="mt-2 text-sm leading-relaxed text-ink-2">
            <LicenseLine />
          </p>
        </div>
      </div>
      <div className="border-t border-line">
        <p className="container-site py-5 text-[0.75rem] leading-relaxed text-ink-3">
          The GridWorld on this page is a TypeScript port of RLForge&apos;s algorithm and config, not the C++ library compiled for the web.
          Star and fork counts come live from the GitHub API, refreshed at most hourly.
        </p>
      </div>
    </footer>
  );
}
