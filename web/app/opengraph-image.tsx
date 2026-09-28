import { ImageResponse } from "next/og";
import { TAGLINE, TOTAL_TESTS } from "@/lib/content";

export const alt = "RLForge: We built the tensor engine so we didn't have to trust anyone else's gradients.";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Same geometry as app/icon.svg (the optimal GridWorld route, steel to ember).
const MARK = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><defs><linearGradient id="p" x1="7" y1="25" x2="25" y2="7" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#80c4ff"/><stop offset="0.55" stop-color="#ff8634"/><stop offset="1" stop-color="#ffb84c"/></linearGradient><radialGradient id="g" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="#ffd896"/><stop offset="0.45" stop-color="#ff8634"/><stop offset="1" stop-color="#ff5c26" stop-opacity="0"/></radialGradient></defs><rect x="1" y="1" width="30" height="30" rx="8.5" fill="#15171b"/><rect x="1.5" y="1.5" width="29" height="29" rx="8" fill="none" stroke="#383d48"/><path d="M7.5 24.5H12V20h4.5v-4.5H21V11h3.5" fill="none" stroke="url(#p)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/><circle cx="24.5" cy="7.5" r="5" fill="url(#g)"/><circle cx="24.5" cy="7.5" r="2.2" fill="#fff1d6"/></svg>`;

export default function OpengraphImage() {
  const chips = ["C++20", `${TOTAL_TESTS} tests`, "zero dependencies", "-Werror"];
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px 80px",
          backgroundColor: "#0b0c0e",
          backgroundImage:
            "radial-gradient(circle at 85% 18%, rgba(255,92,38,0.30) 0%, rgba(11,12,14,0) 45%), radial-gradient(circle at 0% 100%, rgba(86,160,232,0.20) 0%, rgba(11,12,14,0) 40%)",
          color: "#eef0f3",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 22 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`data:image/svg+xml;base64,${Buffer.from(MARK).toString("base64")}`} width={92} height={92} alt="" />
          <div style={{ display: "flex", fontSize: 58, fontWeight: 700, letterSpacing: -2 }}>
            <span>RL</span>
            <span style={{ color: "#ff8634" }}>Forge</span>
          </div>
        </div>
        <div style={{ display: "flex", fontSize: 64, fontWeight: 700, lineHeight: 1.05, letterSpacing: -2, maxWidth: 1040 }}>{TAGLINE}</div>
        <div style={{ display: "flex", gap: 14 }}>
          {chips.map((c) => (
            <div
              key={c}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 20px",
                borderRadius: 999,
                border: "1px solid #383d48",
                background: "#1c1f25",
                color: "#c3c8d1",
                fontSize: 26,
              }}
            >
              <div style={{ width: 10, height: 10, borderRadius: 999, background: "#ff5c26" }} />
              {c}
            </div>
          ))}
        </div>
      </div>
    ),
    size,
  );
}
