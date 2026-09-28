import { NextResponse } from "next/server";
import { getRepoStats } from "@/lib/github";

// The client calls this to refresh the star/fork counts the page was
// server-rendered with. The route itself runs per request, but the GitHub
// fetch inside getRepoStats() is held in Next's data cache for an hour, the
// same ISR window as the page, so GitHub sees at most one request per hour.
export const dynamic = "force-dynamic";

export async function GET() {
  const stats = await getRepoStats();
  if (!stats) return NextResponse.json({ error: "unavailable" }, { status: 503 });
  return NextResponse.json(stats);
}
