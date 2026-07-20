import { el } from "../dom.js";
import { inferno } from "../heat.js";

export interface BootScreen {
  element: HTMLElement;
  /** Target focus percent [0..100]; the pull eases toward it. */
  setFocus(pct: number): void;
  /** Scrambled status line (a phase label). No-op if unchanged. */
  setStatus(text: string): void;
  /** Error state: red status + optional faint hint line. */
  fail(text: string, hint?: string): void;
  /** Drive to 100, clamp the reticle, hold briefly. */
  lock(): Promise<void>;
  /** Aperture-out animation, then done. */
  exit(): Promise<void>;
}

const GRID_N = 12;
const SCRAMBLE_CHARS = "!-_/[]{}=+*^?#01·:";
const GRAIN_SVG =
  '<svg width="100%" height="100%" preserveAspectRatio="none">' +
  '<filter id="vf-noise"><feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="2" stitchTiles="stitch"/>' +
  '<feColorMatrix type="saturate" values="0"/></filter>' +
  '<rect width="100%" height="100%" filter="url(#vf-noise)"/></svg>';

const clamp = (v: number, a: number, b: number): number => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Viewfinder boot screen: a 12x12 grid pulls into focus as the engine loads
 *  (focus % tracks real progress), then apertures out to reveal the app. */
export function createBootScreen(): BootScreen {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const mark = el("div", { class: "vf-mark" });
  [..."fovea"].forEach((ch, i) => {
    const s = el("span", {}, [ch]);
    s.style.animationDelay = `${i * 72}ms`;
    mark.append(s);
  });

  const grid = el("div", { class: "vf-grid" });
  const cells: HTMLElement[] = [];
  const ordering: { c: HTMLElement; o: number }[] = [];
  for (let y = 0; y < GRID_N; y++) {
    for (let x = 0; x < GRID_N; x++) {
      const c = el("i");
      cells.push(c);
      ordering.push({ c, o: x + y + Math.random() * 0.85 });
      grid.append(c);
    }
  }
  const sorted = ordering.sort((a, b) => a.o - b.o).map((e) => e.c);
  // color each cell by ignition order for a smooth heatmap gradient
  sorted.forEach((c, r) => c.style.setProperty("--c", inferno(0.18 + 0.82 * (r / (sorted.length - 1)))));

  const reticle = el("div", { class: "vf-reticle" }, [
    el("i", { class: "tl" }), el("i", { class: "tr" }),
    el("i", { class: "bl" }), el("i", { class: "br" }),
  ]);
  const finder = el("div", { class: "vf-finder" }, [reticle, grid]);

  const countEl = el("span", {}, ["000"]);
  const fillEl = el("div", { class: "vf-fill" });
  const statusEl = el("div", { class: "vf-status" }, [" "]);
  const hintEl = el("div", { class: "vf-hint" });

  const stage = el("div", { class: "vf-stage" }, [
    mark,
    el("div", { class: "vf-sub" }, ["forward observation · vector editing of activations"]),
    finder,
    el("div", { class: "vf-readout" }, [
      el("div", { class: "vf-focusrow" }, [
        el("span", { class: "vf-flabel" }, ["focus"]),
        el("span", { class: "vf-count" }, [countEl, "%"]),
      ]),
      el("div", { class: "vf-meter" }, [fillEl]),
      statusEl,
      hintEl,
    ]),
  ]);

  const grain = el("div", { class: "vf-grain", "aria-hidden": "true" });
  grain.innerHTML = GRAIN_SVG;

  const element = el("div", { class: "boot-screen", role: "status", "aria-label": "Loading" }, [
    el("div", { class: "vf-vig" }),
    grain,
    el("div", { class: "vf-hud bl" }, ["webgpu · f16"]),
    el("div", { class: "vf-hud br" }, ["gpt-2 · 124m"]),
    stage,
  ]);

  let focus = 0, target = 0, locked = false, disposed = false, raf = 0;

  const lightGrid = (): void => {
    const n = Math.round((focus / 100) * sorted.length);
    sorted.forEach((c, r) => {
      c.classList.toggle("lit", r < n);
      c.classList.toggle("active", !locked && r >= n - 7 && r < n);
    });
  };
  const render = (): void => {
    countEl.textContent = String(Math.floor(focus)).padStart(3, "0");
    fillEl.style.width = `${focus}%`;
    element.style.setProperty("--vf-spread", `${lerp(34, 10, focus / 100)}px`);
    if (!reduce) grid.style.filter = `blur(${lerp(16, 0.3, focus / 100).toFixed(2)}px)`;
    lightGrid();
  };
  const loop = (): void => {
    focus += (target - focus) * 0.06;
    if (Math.abs(target - focus) < 0.15) focus = target;
    render();
    if (!disposed) raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  const scrambleRaf = new WeakMap<HTMLElement, number>();
  let lastStatus = "";
  const scramble = (elm: HTMLElement, text: string): void => {
    if (reduce) { elm.textContent = text; return; }
    const old = (elm.textContent || "").replace(/ /g, "");
    const len = Math.max(old.length, text.length);
    const q = Array.from({ length: len }, (_, i) => ({
      from: old[i] || "", to: text[i] || "",
      start: Math.floor(Math.random() * 10), end: Math.floor(Math.random() * 10) + 10, ch: "",
    }));
    const prev = scrambleRaf.get(elm);
    if (prev) cancelAnimationFrame(prev);
    let frame = 0;
    const step = (): void => {
      let out = "", done = 0;
      for (const it of q) {
        if (frame >= it.end) { done++; out += it.to; }
        else if (frame >= it.start) {
          if (!it.ch || Math.random() < 0.3) it.ch = SCRAMBLE_CHARS[Math.floor(Math.random() * SCRAMBLE_CHARS.length)];
          out += `<span class="dud">${it.ch}</span>`;
        } else out += it.from;
      }
      elm.innerHTML = out;
      if (done !== q.length) { frame++; scrambleRaf.set(elm, requestAnimationFrame(step)); }
    };
    step();
  };

  const onMove = (e: MouseEvent): void => {
    if (disposed) return;
    finder.style.transform = `translate(${(e.clientX / innerWidth - 0.5) * 10}px, ${(e.clientY / innerHeight - 0.5) * 10}px)`;
  };
  window.addEventListener("mousemove", onMove);

  return {
    element,
    setFocus(pct) { target = clamp(pct, 0, 100); },
    setStatus(text) {
      if (text === lastStatus) return;
      lastStatus = text;
      statusEl.classList.remove("error");
      scramble(statusEl, text);
    },
    fail(text, hint) {
      lastStatus = text;
      statusEl.classList.add("error");
      scramble(statusEl, text);
      if (hint) { hintEl.textContent = hint; hintEl.classList.add("show"); }
    },
    async lock() {
      target = 100;
      const t0 = performance.now();
      while (focus < 99.4 && performance.now() - t0 < 2200) await sleep(30);
      focus = 100;
      locked = true;
      element.classList.add("locked");
      element.style.setProperty("--vf-spread", "9px");
      grid.style.filter = "none";
      render();
      this.setStatus("focus locked · 8/8 prompts match");
      await sleep(780);
    },
    async exit() {
      element.classList.add("exit");
      element.style.setProperty("--vf-spread", "50px");
      await sleep(reduce ? 0 : 640);
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("mousemove", onMove);
    },
  };
}