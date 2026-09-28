"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { RepoStats } from "@/lib/github";
import { StarIcon } from "./Icons";

const Ctx = createContext<RepoStats | null>(null);

/**
 * Hydrates with the server-rendered (hourly ISR) stats, then refreshes once
 * from /api/repo. If both fail, consumers get null and show no count.
 */
export function RepoStatsProvider({ initial, children }: { initial: RepoStats | null; children: React.ReactNode }) {
  const [stats, setStats] = useState(initial);
  useEffect(() => {
    const ctrl = new AbortController();
    fetch("/api/repo", { signal: ctrl.signal })
      .then((r) => (r.ok ? (r.json() as Promise<RepoStats>) : null))
      .then((s) => {
        if (s && typeof s.stars === "number") setStats(s);
      })
      .catch(() => {});
    return () => ctrl.abort();
  }, []);
  return <Ctx.Provider value={stats}>{children}</Ctx.Provider>;
}

export const useRepoStats = () => useContext(Ctx);

const fmt = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

function fetchedLabel(iso: string) {
  const d = new Date(iso);
  return `Live from the GitHub API, fetched ${d.toUTCString().replace(/:\d\d GMT$/, " UTC")}`;
}

export function StarCount({ className }: { className?: string }) {
  const stats = useRepoStats();
  if (!stats) return null;
  return (
    <span className={className} title={fetchedLabel(stats.fetchedAt)}>
      <StarIcon className="h-3 w-3" />
      <span className="sr-only">GitHub stars: </span>
      {fmt.format(stats.stars)}
    </span>
  );
}

export function ForkCount({ className }: { className?: string }) {
  const stats = useRepoStats();
  if (!stats) return null;
  return (
    <span className={className} title={fetchedLabel(stats.fetchedAt)}>
      {fmt.format(stats.forks)} <span className="text-ink-3">{stats.forks === 1 ? "fork" : "forks"}</span>
    </span>
  );
}

export function LicenseLine() {
  const stats = useRepoStats();
  // The README declares MIT but, at the time this site was written, the repo
  // had no LICENSE file. Only link one when GitHub actually detects it.
  if (stats?.licenseUrl) {
    return (
      <a className="link" href={stats.licenseUrl}>
        {stats.license ?? "License"} license
      </a>
    );
  }
  return <span>MIT, as declared in the README. The LICENSE file itself hasn&apos;t landed in the repo yet.</span>;
}
