"use client";

import { useEffect } from "react";

/**
 * Page-wide progressive enhancements:
 *  - scroll reveals for [data-reveal] (content is visible without JS; see globals.css)
 *  - the cursor-aware hover light on .spot elements (fine pointers only)
 */
export function ClientEffects() {
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canObserve = !reduced && typeof IntersectionObserver !== "undefined";
    const io = canObserve
      ? new IntersectionObserver(
          (entries) => {
            for (const e of entries) {
              if (e.isIntersecting) {
                e.target.classList.add("is-in");
                io!.unobserve(e.target);
              }
            }
          },
          { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
        )
      : null;

    const track = (root: ParentNode) => {
      root.querySelectorAll<HTMLElement>("[data-reveal]:not(.is-in)").forEach((el) => {
        if (io) io.observe(el);
        else el.classList.add("is-in");
      });
    };
    track(document);

    // React can replace DOM nodes after this effect runs (client re-renders,
    // Fast Refresh), so pick up any [data-reveal] element added later too.
    const mo = new MutationObserver((records) => {
      for (const r of records) {
        r.addedNodes.forEach((n) => {
          if (!(n instanceof HTMLElement)) return;
          if (n.matches("[data-reveal]:not(.is-in)")) {
            if (io) io.observe(n);
            else n.classList.add("is-in");
          }
          track(n);
        });
      }
    });
    mo.observe(document.body, { childList: true, subtree: true });

    const fine = window.matchMedia("(hover: hover) and (pointer: fine)");
    let raf = 0;
    let last: PointerEvent | null = null;
    const onMove = (e: PointerEvent) => {
      if (!fine.matches) return;
      last = e;
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const target = (last?.target as Element | null)?.closest?.(".spot") as HTMLElement | null;
        if (!target || !last) return;
        const r = target.getBoundingClientRect();
        target.style.setProperty("--mx", `${last.clientX - r.left}px`);
        target.style.setProperty("--my", `${last.clientY - r.top}px`);
      });
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    return () => {
      io?.disconnect();
      mo.disconnect();
      window.removeEventListener("pointermove", onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  return null;
}
