import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import { ClientEffects } from "@/components/ClientEffects";
import { TAGLINE } from "@/lib/content";
import "./globals.css";

const sans = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });
const serif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  variable: "--font-serif",
  display: "swap",
});

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000");

const description =
  "A from-scratch reinforcement-learning library in C++20: its own tensor and autograd engine, NN layers, optimizers, tabular Q-learning, DQN and PPO. No third-party dependencies beyond Catch2 for the tests.";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "RLForge: reinforcement learning, forged from scratch in C++20", template: "%s · RLForge" },
  description,
  openGraph: { title: "RLForge", description: TAGLINE, type: "website", siteName: "RLForge" },
  twitter: { card: "summary_large_image", title: "RLForge", description: TAGLINE },
};

export const viewport: Viewport = {
  themeColor: "#0b0c0e",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`} suppressHydrationWarning>
      <head>
        {/* Marks JS as available before first paint so scroll reveals never hide content without JS. */}
        <script dangerouslySetInnerHTML={{ __html: "document.documentElement.classList.add('js')" }} />
      </head>
      <body>
        <div className="grain" aria-hidden />
        {children}
        <ClientEffects />
      </body>
    </html>
  );
}
