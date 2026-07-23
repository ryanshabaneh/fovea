import { el, replace } from "../dom.js";
import { inferno } from "../heat.js";
import { fmtProb, display } from "../format.js";

const LAYERS = 12;
const HEADS = 12;

export interface HeadAttention {
  /** Context tokens with the selected head's last-token attention weight [0..1]. */
  tokens: { text: string; weight: number }[];
}

export interface HeadDetail {
  layer: number;
  head: number;
  kl: number;
  importance: string;
  cleanTop: string;
  ablatedTop: string;
  shifts: { token: string; from: number; to: number }[];
  attention?: HeadAttention | null;
}

export interface HeatmapScene {
  element: HTMLElement;
  onHead(fn: (layer: number, head: number) => void): void;
  setCell(layer: number, head: number, intensity: number): void;
  clearCells(): void;
  markExplored(layer: number, head: number): void;
  setSelected(layer: number, head: number): void;
  setCaption(text: string): void;
  setEnabled(enabled: boolean): void;
  setDetail(d: HeadDetail | null): void;
}

const axis = (cls: string): HTMLElement =>
  el("div", { class: cls }, Array.from({ length: 12 }, (_, i) => el("span", {}, [String(i)])));

/** The selected head's last-token attention, painted onto the context tokens.
 *  Weights are normalized to the row's peak and gamma-lifted so that secondary
 *  attention stays visible on the dark colormap. */
function attentionSection(tokens: { text: string; weight: number }[]): HTMLElement {
  let peak = 0, peakIdx = 0;
  tokens.forEach((t, i) => { if (t.weight > peak) { peak = t.weight; peakIdx = i; } });
  const denom = peak || 1;

  const chips = tokens.map((t) => {
    const g = Math.pow(Math.min(1, t.weight / denom), 0.6);
    const chip = el("span", { class: "attn-tok", title: `${(t.weight * 100).toFixed(1)}%` }, [t.text]);
    chip.style.background = inferno(g);
    chip.style.color = g > 0.52 ? "#0d0d0f" : "var(--text-dim)";
    return chip;
  });

  return el("div", { class: "attn-sec" }, [
    el("div", { class: "hm-d-eyebrow" }, ["attends to"]),
    el("div", { class: "attn-strip" }, chips),
    el("div", { class: "attn-peak" }, ["max ", el("b", {}, [tokens[peakIdx].text]), ` · ${(peak * 100).toFixed(0)}%`]),
  ]);
}

/** Attention-head ablation importance, ΔKL from the clean prediction. */
export function createHeatmapScene(): HeatmapScene {
  const cells: HTMLElement[] = [];
  const grid = el("div", { class: "hm-grid" });
  for (let l = 0; l < LAYERS; l++) {
    for (let h = 0; h < HEADS; h++) {
      const c = el("i", { "data-l": String(l), "data-h": String(h), title: `L${l} · H${h}` });
      cells.push(c);
      grid.append(c);
    }
  }
  const idx = (l: number, h: number): number => l * HEADS + h;

  const caption = el("div", { class: "hm-caption" }, ["scanning heads…"]);
  const detail = el("div", { class: "hm-detail" });

  const element = el("div", { class: "scene hm-scene" }, [
    el("div", { class: "hm-wrap" }, [
      el("div", { class: "hm-head" }, [
        el("h1", { class: "hm-title" }, ["Attention-head ablation"]),
        el("div", { class: "hm-note" }, ["Each cell is an attention head, colored by the KL divergence between the next-token distribution with that head intact versus ablated. Brighter heads exert more influence on the prediction."]),
      ]),
      el("div", { class: "hm-main" }, [
        el("div", { class: "hm-spacer" }),
        el("div", { class: "hm-plot" }, [
          el("div", { class: "hm-ax-h-title" }, ["heads"]),
          el("div", { class: "hm-heads-ax" }, [axis("hm-ax-h")]),
          el("div", { class: "hm-row" }, [
            el("div", { class: "hm-ax-l-title" }, ["layers"]),
            axis("hm-ax-l"),
            grid,
          ]),
          el("div", { class: "hm-legend" }, [el("span", {}, ["0"]), el("span", { class: "hm-ramp" }), el("span", {}, ["max"])]),
          caption,
        ]),
        detail,
      ]),
    ]),
  ]);

  const idleDetail = (): void => { replace(detail, el("div", { class: "hm-d-idle" }, ["Select a head."])); };
  idleDetail();

  let selected = -1;
  const clearSel = (): void => { if (selected >= 0) cells[selected].classList.remove("sel"); selected = -1; };

  return {
    element,
    onHead(fn) {
      grid.addEventListener("click", (e) => {
        const t = (e.target as HTMLElement).closest("i");
        if (!t || t.parentElement !== grid) return;
        fn(Number(t.getAttribute("data-l")), Number(t.getAttribute("data-h")));
      });
    },
    setCell(l, h, intensity) {
      const c = cells[idx(l, h)];
      c.style.setProperty("--c", inferno(intensity));
      c.classList.add("lit");
    },
    clearCells() {
      for (const c of cells) { c.classList.remove("lit", "explored", "sel"); c.style.removeProperty("--c"); }
      selected = -1;
      idleDetail();
    },
    markExplored(l, h) { cells[idx(l, h)].classList.add("explored"); },
    setSelected(l, h) {
      clearSel();
      if (l >= 0 && h >= 0) { selected = idx(l, h); cells[selected].classList.add("sel"); }
    },
    setCaption(text) { replace(caption, text); },
    setEnabled(enabled) { grid.classList.toggle("disabled", !enabled); },
    setDetail(d) {
      if (!d) { idleDetail(); return; }
      const shifts = d.shifts.map((s) => {
        const dir = s.to > s.from + 1e-4 ? "up" : s.to < s.from - 1e-4 ? "down" : "flat";
        return el("div", { class: "shift-row" }, [
          el("span", { class: "shift-tok" }, [display(s.token)]),
          el("span", { class: "shift-from" }, [fmtProb(s.from)]),
          el("span", { class: "shift-arrow" }, ["→"]),
          el("span", { class: `shift-to ${dir}` }, [fmtProb(s.to)]),
        ]);
      });
      const flipped = d.cleanTop !== d.ablatedTop;
      const predChildren: (Node | string)[] = ["prediction ", el("b", {}, [display(d.cleanTop)])];
      if (flipped) predChildren.push(el("span", { class: "hm-d-arrow" }, [" → "]), el("b", { class: "hm-d-flip" }, [display(d.ablatedTop)]));
      else predChildren.push(el("span", { class: "hm-d-unch" }, [" · unchanged"]));

      const shiftHead = el("div", { class: "shift-row shift-head" }, [
        el("span", {}, ["token"]),
        el("span", {}, ["clean"]),
        el("span", {}, [""]),
        el("span", {}, ["ablated"]),
      ]);

      const nodes: (Node | string)[] = [
        el("div", { class: "hm-d-head" }, [`Layer ${d.layer} · Head ${d.head}`]),
        el("div", { class: "hm-d-kl" }, [`Δ KL ${d.kl.toFixed(3)} · `, el("span", { class: `hm-d-imp imp-${d.importance}` }, [d.importance])]),
      ];
      if (d.attention && d.attention.tokens.length) nodes.push(attentionSection(d.attention.tokens));
      nodes.push(
        el("div", { class: "hm-d-eyebrow" }, ["if ablated"]),
        el("div", { class: "hm-d-pred" }, predChildren),
        shiftHead,
        el("div", { class: "hm-d-shifts" }, shifts),
      );
      replace(detail, ...nodes);
    },
  };
}