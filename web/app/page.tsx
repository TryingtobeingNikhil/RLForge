import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Footer } from "@/components/Footer";
import { RepoStatsProvider } from "@/components/RepoStatsProvider";
import { DiamondSection } from "@/components/sections/DiamondSection";
import { RulesSection } from "@/components/sections/RulesSection";
import { ProofSection } from "@/components/sections/ProofSection";
import { ArchitectureSection } from "@/components/sections/ArchitectureSection";
import { BuildLogSection } from "@/components/sections/BuildLogSection";
import { TestsSection } from "@/components/sections/TestsSection";
import { ScarsSection } from "@/components/sections/ScarsSection";
import { QuickstartSection } from "@/components/sections/QuickstartSection";
import { getRepoStats } from "@/lib/github";

// Hourly ISR: the page (and its GitHub numbers) re-render at most once an hour.
export const revalidate = 3600;

export default async function Page() {
  const stats = await getRepoStats();
  return (
    <RepoStatsProvider initial={stats}>
      <a href="#diamond" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[70] focus:rounded-full focus:bg-surface-2 focus:px-4 focus:py-2 focus:text-ink">
        Skip past the demo
      </a>
      <Nav />
      <main>
        <Hero />
        <DiamondSection />
        <RulesSection />
        <ProofSection />
        <ArchitectureSection />
        <BuildLogSection />
        <TestsSection />
        <ScarsSection />
        <QuickstartSection />
      </main>
      <Footer />
    </RepoStatsProvider>
  );
}
