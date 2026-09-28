// A tiny, dependency-free highlighter for the C++ and shell snippets on this
// page. It only needs to be pleasant, not a parser.
import type { ReactNode } from "react";

export type Lang = "cpp" | "bash" | "text";

const CPP_KEYWORDS = new Set([
  "auto", "const", "return", "using", "namespace", "for", "if", "else", "true", "false", "struct", "class",
  "void", "int", "float", "double", "bool", "size_t", "int64_t", "uint64_t", "try", "catch", "throw", "new",
  "public", "virtual", "override", "final", "static", "std", "nullptr", "template", "typename",
]);

interface Rule {
  re: RegExp;
  cls: string | ((m: string) => string);
  /** Only match at the start of a line or right after `&& `. */
  commandPosition?: boolean;
}

const RULES: Record<Exclude<Lang, "text">, Rule[]> = {
  cpp: [
    { re: /^\/\/[^\n]*/, cls: "text-ink-3 italic" },
    { re: /^#\s*include[^\n]*/, cls: "text-steel" },
    { re: /^"(?:[^"\\\n]|\\.)*"/, cls: "text-amber" },
    { re: /^\d+(?:\.\d+)?f?\b/, cls: "text-ember-2" },
    { re: /^[A-Za-z_]\w*/, cls: (w) => (CPP_KEYWORDS.has(w) ? "text-[#ff9d7a]" : /^[A-Z]/.test(w) ? "text-steel" : "") },
  ],
  bash: [
    { re: /^#[^\n]*/, cls: "text-ink-3 italic" },
    { re: /^(?:git|cmake|cd|ctest|npm|clang\+\+)(?=\s|$)/, cls: "text-steel", commandPosition: true },
    { re: /^-{1,2}[\w-]+(?:=\S*)?/, cls: "text-amber" },
    { re: /^\$\([^)\n]*\)/, cls: "text-ember-2" },
  ],
};

export function highlight(code: string, lang: Lang): ReactNode[] {
  if (lang === "text") return [code];
  const rules = RULES[lang];
  const out: ReactNode[] = [];
  let plain = "";
  let key = 0;
  let i = 0;
  while (i < code.length) {
    const rest = code.slice(i);
    const before = code.slice(0, i);
    const prev = before.at(-1) ?? "\n";
    let hit: { text: string; cls: string } | null = null;
    for (const rule of rules) {
      const m = rule.re.exec(rest);
      if (!m) continue;
      if (/^\w/.test(m[0]) && /\w/.test(prev)) continue; // don't match inside a word
      if (rule.commandPosition && !(prev === "\n" || /&&\s+$/.test(before))) continue;
      hit = { text: m[0], cls: typeof rule.cls === "function" ? rule.cls(m[0]) : rule.cls };
      break;
    }
    if (hit) {
      if (plain) out.push(plain);
      plain = "";
      out.push(hit.cls ? <span key={key++} className={hit.cls}>{hit.text}</span> : hit.text);
      i += hit.text.length;
    } else {
      plain += code[i];
      i++;
    }
  }
  if (plain) out.push(plain);
  return out;
}
