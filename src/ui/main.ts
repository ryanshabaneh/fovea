import "./styles.css";
import { boot, type BootProgress } from "../engine/boot.js";
import { createBootScreen } from "./components/bootscreen.js";
import { createTopBar } from "./components/topbar.js";
import { createConsoleScene, type DistTok } from "./scenes/console.js";
import { createHeatmapScene } from "./scenes/heatmap.js";
import { HeadAblation } from "../engine/interventions/ablation.js";
import { softmaxStats, softmaxProbs, klDiv, topK, prob, argmax } from "./format.js";
import { GPT2_SMALL } from "../engine/config.js";

const app = document.getElementById("app")!;
const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const V = GPT2_SMALL.vocab_size;
const L = GPT2_SMALL.n_layers;
const H = GPT2_SMALL.n_heads;
const BOS_ID = 50256;

let fwd: Awaited<ReturnType<typeof boot>>["fwd"] | null = null;
let tokenizer: Awaited<ReturnType<typeof boot>>["tokenizer"] | null = null;

interface Baseline { tokens: Uint32Array; probs: Float32Array; top: DistTok[]; }
let baseline: Baseline | null = null;
let sweep: Float32Array | null = null;
let sweepMax = 0;
const explored = new Set<number>();
let busy = false;

const bootScreen = createBootScreen();
const topbar = createTopBar();
const con = createConsoleScene("The Eiffel Tower is in the city of");
const heatmap = createHeatmapScene();

let current: HTMLElement | null = null;
function show(next: HTMLElement): void {
  if (next === current) return;
  const prev = current;
  next.classList.remove("scene-out");
  next.classList.add("scene");
  app.appendChild(next);
  requestAnimationFrame(() => {
    next.classList.add("scene-in");
    if (prev) prev.classList.add("scene-out");
  });
  if (prev && prev !== next) setTimeout(() => { if (prev !== current) prev.remove(); }, reduce ? 0 : 440);
  current = next;
}

const decode = (id: number): string => tokenizer!.decode([id]);
const importanceOf = (kl: number): string => (kl > 0.25 ? "high" : kl > 0.06 ? "moderate" : "low");

function topTokens(lastRow: Float32Array, n: number): DistTok[] {
  const stats = softmaxStats(lastRow);
  return topK(lastRow, n).map((t) => ({ id: t.id, token: decode(t.id), logit: t.logit, prob: prob(t.logit, stats) }));
}

function encodeInput(): Uint32Array {
  const ids = tokenizer!.encode(con.getPrompt());
  return con.getBos() ? Uint32Array.from([BOS_ID, ...ids]) : ids;
}

function refreshTokens(): void {
  if (!tokenizer) return;
  con.setTokens(Array.from(encodeInput()).map((id) => ({ id, text: decode(id) })));
}

async function ablate(layer: number, headIdx: number): Promise<Float32Array> {
  const { logits, seqLen } = await fwd!.run(baseline!.tokens, { writes: [new HeadAblation(layer, headIdx).toHookWrite()] });
  return logits.slice((seqLen - 1) * V, seqLen * V);
}

async function runBaseline(): Promise<void> {
  if (!fwd || !tokenizer || busy) return;
  const tokens = encodeInput();
  if (tokens.length === 0) return;
  busy = true;
  con.setBusy(true);
  topbar.setBusy(true);
  try {
    const t0 = performance.now();
    const { logits, seqLen } = await fwd.run(tokens);
    const ms = performance.now() - t0;
    const lastRow = logits.slice((seqLen - 1) * V, seqLen * V);
    baseline = { tokens, probs: softmaxProbs(lastRow), top: topTokens(lastRow, 6) };
    sweep = null; sweepMax = 0; explored.clear(); heatmap.clearCells();
    con.renderDistribution(baseline.top);
    con.setRunInfo(`${seqLen} tokens · ${ms.toFixed(0)} ms`);
  } finally {
    busy = false;
    con.setBusy(false);
    topbar.setBusy(false);
  }
}

function paintSweep(): void {
  if (!sweep) return;
  for (let i = 0; i < sweep.length; i++) heatmap.setCell(Math.floor(i / H), i % H, Math.min(1, sweep[i] / sweepMax));
  for (const i of explored) heatmap.markExplored(Math.floor(i / H), i % H);
}

function backToConsole(): void { show(con.element); topbar.setNav(null); }

async function enterHeatmap(): Promise<void> {
  show(heatmap.element);
  topbar.setNav("console", backToConsole);
  if (sweep) { paintSweep(); heatmap.setCaption(`peak Δ KL ${sweepMax.toFixed(3)}`); return; }
  if (busy || !baseline) return;
  busy = true;
  topbar.setBusy(true);
  heatmap.setEnabled(false);
  try {
    const b = baseline;
    const kls = new Float32Array(L * H);
    let max = 1e-9;
    for (let l = 0; l < L; l++) {
      for (let h = 0; h < H; h++) {
        const kl = klDiv(b.probs, softmaxProbs(await ablate(l, h)));
        kls[l * H + h] = kl;
        max = Math.max(max, kl);
        heatmap.setCell(l, h, Math.min(1, kl / max));
        heatmap.setCaption(`${l * H + h + 1} / ${L * H}`);
      }
    }
    sweep = kls; sweepMax = max;
    paintSweep();
    heatmap.setCaption(`peak Δ KL ${sweepMax.toFixed(3)}`);
  } finally {
    busy = false;
    topbar.setBusy(false);
    heatmap.setEnabled(true);
  }
}

async function investigateHead(layer: number, headIdx: number): Promise<void> {
  if (!baseline || busy) return;
  busy = true;
  topbar.setBusy(true);
  try {
    const b = baseline;
    const row = await ablate(layer, headIdx);
    const abProbs = softmaxProbs(row);
    const kl = klDiv(b.probs, abProbs);
    const abTopId = argmax(row);

    const shown = b.top.slice(0, 5);
    const shifts = shown.map((t) => ({ token: t.token, from: b.probs[t.id], to: abProbs[t.id] }));
    // append the ablated argmax if it isn't already in the top-5
    if (!shown.some((t) => t.id === abTopId)) {
      shifts.push({ token: decode(abTopId), from: b.probs[abTopId], to: abProbs[abTopId] });
    }

    heatmap.setSelected(layer, headIdx);
    heatmap.setDetail({
      layer, head: headIdx, kl, importance: importanceOf(kl),
      cleanTop: b.top[0].token, ablatedTop: decode(abTopId), shifts,
    });
    explored.add(layer * H + headIdx);
    heatmap.markExplored(layer, headIdx);
  } finally {
    busy = false;
    topbar.setBusy(false);
  }
}

con.onRun(() => void runBaseline());
con.onInput(refreshTokens);
con.onAttribute(() => void enterHeatmap());
heatmap.onHead((l, h) => void investigateHead(l, h));

function onProgress(p: BootProgress): void {
  if (p.stage === "weights" && p.total && p.total > 1) {
    bootScreen.setFocus(10 + (p.loaded! / p.total) * 74);
    bootScreen.setStatus("acquiring signal · weights.bin");
    return;
  }
  switch (p.stage) {
    case "adapter":   bootScreen.setFocus(6);  bootScreen.setStatus("requesting GPU adapter"); break;
    case "weights":   bootScreen.setFocus(9);  bootScreen.setStatus("acquiring signal · weights.bin"); break;
    case "kernels":   bootScreen.setFocus(88); bootScreen.setStatus("compiling wgsl kernels"); break;
    case "tokenizer": bootScreen.setFocus(95); bootScreen.setStatus("resolving tokenizer"); break;
    case "ready":     bootScreen.setFocus(99); bootScreen.setStatus("bringing into focus"); break;
  }
}

async function start(): Promise<void> {
  const t0 = performance.now();
  app.replaceChildren(bootScreen.element);
  bootScreen.setStatus("initializing engine");

  let booted;
  try {
    booted = await boot({ onProgress });
  } catch (e) {
    const msg = (e as Error).message;
    let hint: string | undefined;
    if (/fetch|network|load failed/i.test(msg)) hint = "content blocker or offline · try another browser profile";
    else if (/webgpu|adapter/i.test(msg)) hint = "use Chrome/Edge 113+ with hardware acceleration on";
    bootScreen.fail("boot failed · signal lost", hint);
    return;
  }

  fwd = booted.fwd;
  tokenizer = booted.tokenizer;
  refreshTokens();

  await bootScreen.lock();
  const elapsed = performance.now() - t0;
  if (elapsed < 1800) await sleep(1800 - elapsed);
  await bootScreen.exit();

  bootScreen.element.remove();
  app.replaceChildren(topbar.element);
  current = null;
  show(con.element);
  con.focus();
}

void start();