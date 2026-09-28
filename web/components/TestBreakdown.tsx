"use client";

import { useState } from "react";
import { REPO, TEST_GROUPS, TEST_TAGS, TOTAL_TESTS, groupTotal } from "@/lib/content";

export function TestBreakdown() {
  const [view, setView] = useState<"subsystem" | "tag">("subsystem");
  const [open, setOpen] = useState<string | null>(null);
  const maxGroup = Math.max(...TEST_GROUPS.map(groupTotal));
  const maxTag = TEST_TAGS[0].count;

  return (
    <div className="card p-4 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div role="tablist" aria-label="Group tests by" className="flex rounded-full border border-line-2 bg-bg-2 p-0.5">
          {(
            [
              ["subsystem", "By subsystem"],
              ["tag", "By Catch2 tag"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              role="tab"
              aria-selected={view === id}
              onClick={() => setView(id)}
              className={`rounded-full px-3.5 py-1.5 text-[0.8rem] font-semibold transition-colors ${
                view === id ? "bg-amber/15 text-amber" : "text-ink-3 hover:text-ink"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="font-mono text-[0.7rem] text-ink-3">
          {view === "subsystem" ? `${TEST_GROUPS.length} groups · ${TOTAL_TESTS} TEST_CASEs` : "a test can carry several tags, so these overlap"}
        </p>
      </div>

      {view === "subsystem" ? (
        <ul className="mt-5 space-y-1.5">
          {TEST_GROUPS.map((g) => {
            const n = groupTotal(g);
            const isOpen = open === g.id;
            return (
              <li key={g.id}>
                <button
                  type="button"
                  aria-expanded={isOpen}
                  onClick={() => setOpen(isOpen ? null : g.id)}
                  className="group grid w-full grid-cols-[minmax(0,1fr)_2.5rem] items-center gap-x-3 gap-y-1.5 rounded-xl px-2 py-2 text-left transition-colors hover:bg-surface-2 sm:grid-cols-[15rem_minmax(0,1fr)_3rem]"
                >
                  <span className="truncate text-[0.88rem] text-ink">{g.label}</span>
                  <span className="relative col-span-2 row-start-2 h-2.5 overflow-hidden rounded-full bg-bg-2 sm:col-span-1 sm:col-start-2 sm:row-start-1">
                    <span
                      className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-out"
                      style={{ width: `${(n / maxGroup) * 100}%`, background: g.id === "tensor" ? "var(--hot)" : "rgb(var(--steel) / 0.75)" }}
                    />
                  </span>
                  <span className="col-start-2 row-start-1 text-right font-mono text-sm tabular-nums text-ink sm:col-start-3">{n}</span>
                </button>
                {isOpen ? (
                  <div className="mb-2 ml-2 mt-1 rounded-xl border border-line bg-bg-2/70 px-3 py-2.5">
                    <p className="text-[0.85rem] text-ink-2">{g.blurb}</p>
                    <ul className="mt-2 space-y-0.5 font-mono text-[0.74rem]">
                      {g.files.map((f) => (
                        <li key={f.file} className="flex justify-between gap-3">
                          <a className="link truncate text-ink-2" href={`${REPO.blob}/${f.file}`}>
                            {f.file}
                          </a>
                          <span className="text-ink-3">{f.count}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </li>
            );
          })}
        </ul>
      ) : (
        <ul className="mt-5 flex flex-wrap gap-1.5">
          {TEST_TAGS.map((t) => (
            <li
              key={t.tag}
              className="rounded-full border border-line-2 px-2.5 py-1 font-mono text-[0.74rem]"
              style={{ background: `rgb(var(--steel) / ${0.04 + (t.count / maxTag) * 0.22})` }}
            >
              <span className="text-ink-2">[{t.tag}]</span> <span className="text-ink">{t.count}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
