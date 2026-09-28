// Small inline icons, drawn for this site (no icon library).
type P = { className?: string };

export const GitHubIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="currentColor">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
  </svg>
);

export const StarIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="currentColor">
    <path d="M8 .9l2.1 4.5 4.9.6-3.6 3.4.9 4.9L8 11.9l-4.3 2.4.9-4.9L1 6l4.9-.6L8 .9z" />
  </svg>
);

export const ArrowIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 8h10M9 4l4 4-4 4" />
  </svg>
);

export const PlayIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="currentColor">
    <path d="M4.5 2.8v10.4a.6.6 0 00.9.5l8.3-5.2a.6.6 0 000-1L5.4 2.3a.6.6 0 00-.9.5z" />
  </svg>
);

export const PauseIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="currentColor">
    <rect x="3.5" y="2.5" width="3" height="11" rx="1" />
    <rect x="9.5" y="2.5" width="3" height="11" rx="1" />
  </svg>
);

export const StepIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="currentColor">
    <path d="M3 2.8v10.4a.6.6 0 00.9.5l7-5.2a.6.6 0 000-1l-7-5.2a.6.6 0 00-.9.5z" />
    <rect x="11.5" y="2.5" width="2" height="11" rx="1" />
  </svg>
);

export const ResetIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.8 8a5.2 5.2 0 109-3.6" />
    <path d="M12.3 1.6v3.2H9.1" />
  </svg>
);

export const CopyIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round">
    <rect x="5.5" y="5.5" width="8" height="8" rx="2" />
    <path d="M10.5 3.2V3a1.5 1.5 0 00-1.5-1.5H4A1.5 1.5 0 002.5 3v5A1.5 1.5 0 004 9.5h.3" />
  </svg>
);

export const CheckIcon = ({ className }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 8.5l3 3 7-7" />
  </svg>
);

export const MenuIcon = ({ className, open }: P & { open: boolean }) => (
  <svg viewBox="0 0 20 20" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    {open ? <path d="M5 5l10 10M15 5L5 15" /> : <path d="M3 6h14M3 10h14M3 14h14" />}
  </svg>
);
