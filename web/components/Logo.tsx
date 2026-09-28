import { useId } from "react";

/**
 * The RLForge mark: the optimal GridWorld route, a staircase from the start
 * cell (bottom-left) to the goal (top-right), cooling from steel to ember and
 * ending in a spark. Same geometry as app/icon.svg. The spark only pulses
 * where the mark is always on screen (the sticky nav).
 */
export function LogoMark({ className, title, animated = true }: { className?: string; title?: string; animated?: boolean }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg viewBox="0 0 32 32" className={className} role={title ? "img" : undefined} aria-hidden={title ? undefined : true}>
      {title ? <title>{title}</title> : null}
      <defs>
        <linearGradient id={`${id}p`} x1="7" y1="25" x2="25" y2="7" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#80c4ff" />
          <stop offset="0.55" stopColor="#ff8634" />
          <stop offset="1" stopColor="#ffb84c" />
        </linearGradient>
        <radialGradient id={`${id}g`} cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#ffd896" />
          <stop offset="0.45" stopColor="#ff8634" />
          <stop offset="1" stopColor="#ff5c26" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8.5" fill="#15171b" />
      <rect x="1.5" y="1.5" width="29" height="29" rx="8" fill="none" stroke="#383d48" />
      <path
        d="M7.5 24.5H12V20h4.5v-4.5H21V11h3.5"
        fill="none"
        stroke={`url(#${id}p)`}
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="24.5" cy="7.5" r="5" fill={`url(#${id}g)`} className={animated ? "ember-pulse" : undefined} />
      <circle cx="24.5" cy="7.5" r="2.2" fill="#fff1d6" />
    </svg>
  );
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={className}>
      <span className="font-semibold tracking-[-0.03em] text-ink">RL</span>
      <span className="font-semibold tracking-[-0.03em] text-hot">Forge</span>
    </span>
  );
}
