"use client";

import { useEffect, useState, type RefObject } from "react";

/** True while the element is on screen and the tab is visible. Drives every "pause off-screen". */
export function useInView<T extends Element>(ref: RefObject<T>, rootMargin = "0px"): boolean {
  const [intersecting, setIntersecting] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => setIntersecting(entry.isIntersecting), { rootMargin });
    io.observe(el);
    return () => io.disconnect();
  }, [ref, rootMargin]);

  useEffect(() => {
    const onVis = () => setPageVisible(document.visibilityState === "visible");
    onVis();
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  return intersecting && pageVisible;
}
