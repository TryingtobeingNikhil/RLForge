import { CopyButton } from "./CopyButton";
import { highlight } from "@/lib/highlight";

export function CodeBlock({
  code,
  lang = "cpp",
  title,
  source,
  copy = true,
}: {
  code: string;
  lang?: "cpp" | "bash" | "text";
  title?: string;
  source?: React.ReactNode;
  copy?: boolean;
}) {
  return (
    <figure className="min-w-0">
      {(title || copy) && (
        <div className="mb-2 flex items-center justify-between gap-3">
          {title ? <figcaption className="truncate font-mono text-xs text-ink-3">{title}</figcaption> : <span />}
          {copy ? <CopyButton text={code} /> : null}
        </div>
      )}
      <pre className="code">
        <code>{highlight(code, lang)}</code>
      </pre>
      {source ? <div className="mt-2">{source}</div> : null}
    </figure>
  );
}
