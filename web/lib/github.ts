import { REPO } from "./content";

export interface RepoStats {
  stars: number;
  forks: number;
  /** SPDX id GitHub detected from a license file, or null if the repo has none. */
  license: string | null;
  /** Link to the license file on GitHub, only when one exists. */
  licenseUrl: string | null;
  fetchedAt: string;
}

const API = `https://api.github.com/repos/${REPO.owner}/${REPO.name}`;
export const REVALIDATE_SECONDS = 3600;

function headers(): HeadersInit {
  const h: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "rlforge-web",
  };
  // Optional: raises the unauthenticated 60 requests/hour limit.
  if (process.env.GITHUB_TOKEN) h.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  return h;
}

/**
 * Live repository stats, cached by Next for an hour (ISR). Returns null when
 * GitHub can't be reached, and callers then show no number at all rather
 * than a stale one.
 */
export async function getRepoStats(): Promise<RepoStats | null> {
  try {
    const [repoRes, licenseRes] = await Promise.all([
      fetch(API, { headers: headers(), next: { revalidate: REVALIDATE_SECONDS } }),
      fetch(`${API}/license`, { headers: headers(), next: { revalidate: REVALIDATE_SECONDS } }),
    ]);
    if (!repoRes.ok) return null;
    const repo = (await repoRes.json()) as { stargazers_count?: unknown; forks_count?: unknown };
    if (typeof repo.stargazers_count !== "number" || typeof repo.forks_count !== "number") return null;

    let license: string | null = null;
    let licenseUrl: string | null = null;
    if (licenseRes.ok) {
      const body = (await licenseRes.json()) as { html_url?: string; license?: { spdx_id?: string } };
      license = body.license?.spdx_id && body.license.spdx_id !== "NOASSERTION" ? body.license.spdx_id : null;
      licenseUrl = typeof body.html_url === "string" ? body.html_url : null;
    }

    return {
      stars: repo.stargazers_count,
      forks: repo.forks_count,
      license,
      licenseUrl,
      fetchedAt: new Date().toISOString(),
    };
  } catch {
    return null;
  }
}
