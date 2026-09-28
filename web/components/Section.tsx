export function SectionHeader({
  id,
  eyebrow,
  title,
  aside,
  children,
}: {
  id: string;
  eyebrow: string;
  title: React.ReactNode;
  aside?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <header className="max-w-3xl" data-reveal>
      <p className="eyebrow flex items-center gap-2">
        <span className="h-px w-6 bg-ember" aria-hidden />
        {eyebrow}
      </p>
      <h2 id={`${id}-title`} className="h-section mt-4 text-balance">
        {title}
        {aside ? <span className="aside mt-1 block text-[0.62em] font-normal leading-tight text-ink-2">{aside}</span> : null}
      </h2>
      {children ? <div className="lede mt-5 space-y-4">{children}</div> : null}
    </header>
  );
}

export function Section({ id, children, className = "" }: { id: string; children: React.ReactNode; className?: string }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className={`relative scroll-mt-20 py-20 sm:py-28 ${className}`}>
      <div className="container-site">{children}</div>
    </section>
  );
}

/** A visible source pointer: the file (and line) a claim comes from. */
export function Src({ children, href }: { children: React.ReactNode; href?: string }) {
  const body = (
    <>
      <span aria-hidden className="text-ember">
        ↳
      </span>{" "}
      {children}
    </>
  );
  return href ? (
    <a href={href} className="src inline-block transition-colors hover:text-ink-2">
      {body}
    </a>
  ) : (
    <span className="src inline-block">{body}</span>
  );
}
