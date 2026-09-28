"use client";

import { useEffect, useState } from "react";
import { LogoMark, Wordmark } from "./Logo";
import { GitHubIcon, MenuIcon } from "./Icons";
import { StarCount } from "./RepoStatsProvider";
import { REPO } from "@/lib/content";

const LINKS = [
  { href: "#diamond", label: "Diamond bug" },
  { href: "#rules", label: "Two rules" },
  { href: "#proof", label: "Proof" },
  { href: "#architecture", label: "Architecture" },
  { href: "#build-log", label: "Build log" },
  { href: "#tests", label: "Tests" },
  { href: "#quickstart", label: "Quickstart" },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header
      className={`sticky top-0 z-50 border-b transition-colors duration-300 ${
        scrolled || open ? "border-line bg-bg/85 backdrop-blur-md" : "border-transparent bg-transparent"
      }`}
    >
      <nav className="container-site flex h-16 items-center justify-between gap-4" aria-label="Main">
        <a href="#top" className="group flex items-center gap-2.5 text-lg" aria-label="RLForge home">
          <LogoMark className="h-8 w-8 transition-transform duration-500 ease-spring group-hover:-rotate-6 group-hover:scale-110" />
          <Wordmark />
        </a>

        <ul className="hidden items-center gap-1 lg:flex">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a
                href={l.href}
                className="rounded-full px-3 py-1.5 text-sm text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-2">
          <a
            href={REPO.url}
            className="btn-ghost !px-3.5 !py-2 text-[0.8rem]"
            aria-label="RLForge on GitHub"
          >
            <GitHubIcon className="h-4 w-4" />
            <span className="hidden sm:inline">GitHub</span>
            <StarCount className="inline-flex items-center gap-1 border-l border-line-2 pl-2 font-mono text-xs text-amber" />
          </a>
          <button
            type="button"
            className="ctl !px-2.5 lg:hidden"
            aria-expanded={open}
            aria-controls="mobile-menu"
            aria-label={open ? "Close menu" : "Open menu"}
            onClick={() => setOpen((v) => !v)}
          >
            <MenuIcon open={open} className="h-5 w-5" />
          </button>
        </div>
      </nav>

      <div id="mobile-menu" hidden={!open} className="border-t border-line lg:hidden">
        <ul className="container-site grid grid-cols-2 gap-1 py-3">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a
                href={l.href}
                onClick={() => setOpen(false)}
                className="block rounded-xl px-3 py-2.5 text-[0.95rem] text-ink-2 hover:bg-surface-2 hover:text-ink"
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </header>
  );
}
