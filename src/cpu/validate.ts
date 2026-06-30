/**
 * Validation harness. Node-only this file may import
 * node:fs; nothing under src/engine/ may.
 *
 * Three comparisons share one format and one compare function:
 *   1. CI:        CPU reference  vs TransformerLens golden fixtures (atol 1e-4)
 *   2. Browser:   WGSL path      vs CPU reference (per-kernel tolerances)
 *   3. Browser:   WGSL path      vs golden fixtures (end-to-end atol 1e-2)
 *
 * Tensor-dump format (used by fixtures/ AND by browser GPU dumps):
 *   <dir>/index.json   — { prompts: [{id, text, tokens}],
 *                          tensors: [{prompt, hook, shape, dtype:"f32", file}] }
 *   <dir>/<file>.bin   — raw little-endian f32, C-contiguous
 * Golden tensors keep TL's leading batch dim of 1; comparison squeezes it.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { forward, type CpuWeights } from "./reference.js";
import type { ModelConfig } from "../engine/config.js";

// Core comparison

export interface CompareResult {
  pass: boolean;
  maxAbs: number;       // max |a - b|
  maxRel: number;       // max |a - b| / (|b| + 1e-12)
  worstIndex: number;   // flat index of worst absolute element
  count: number;        // elements compared
}

export interface CompareOptions {
  atol: number;
  rtol?: number;
  /** For hook_attn_scores / hook_pattern: compare only k_pos <= q_pos.
   *  Requires shape [H, T, T] (after batch squeeze). TL versions disagree on
   *  the fill value at masked positions; the lower triangle is the contract. */
  causalLowerOnly?: { H: number; T: number };
}

/** Pass rule,  max|a−b| ≤ atol + rtol·max|b|. */
export function compareTensors(
  a: Float32Array, b: Float32Array, opts: CompareOptions,
): CompareResult {
  if (a.length !== b.length) {
    throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  }
  const rtol = opts.rtol ?? 0;
  let maxAbs = 0, maxRel = 0, worstIndex = -1, maxB = 0, count = 0;

  const include = (i: number): boolean => {
    if (!opts.causalLowerOnly) return true;
    const { T } = opts.causalLowerOnly;
    const k = i % T;
    const q = ((i / T) | 0) % T;
    return k <= q;
  };

  for (let i = 0; i < a.length; i++) {
    if (!include(i)) continue;
    count++;
    const absB = Math.abs(b[i]);
    if (absB > maxB) maxB = absB;
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) { maxAbs = d; worstIndex = i; }
    const rel = d / (absB + 1e-12);
    if (rel > maxRel) maxRel = rel;
  }
  return { pass: maxAbs <= opts.atol + rtol * maxB, maxAbs, maxRel, worstIndex, count };
}

// Dump loading

export interface DumpPrompt { id: string; text: string; tokens: number[]; }
export interface DumpTensorEntry {
  prompt: string; hook: string; shape: number[]; dtype: "f32"; file: string;
}
export interface DumpIndex { prompts: DumpPrompt[]; tensors: DumpTensorEntry[]; }

export function loadIndex(dir: string): DumpIndex {
  return JSON.parse(readFileSync(join(dir, "index.json"), "utf8")) as DumpIndex;
}

export function loadTensor(dir: string, entry: DumpTensorEntry): Float32Array {
  const buf = readFileSync(join(dir, entry.file));
  const expected = entry.shape.reduce((x, y) => x * y, 1);
  const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (arr.length !== expected) {
    throw new Error(`${entry.file}: ${arr.length} elements, shape says ${expected}`);
  }
  return arr;
}

/** Squeeze TL's leading batch dim: [1, ...] → [...]. */
export function squeezeBatch(shape: number[]): number[] {
  return shape[0] === 1 ? shape.slice(1) : shape;
}

// validateAgainstGPU — the function the browser test page reports through.
// The browser writes its readbacks using the same dump format into
// dumps/gpu/<promptId>/, then this runs in Node (or the comparison runs
// directly in-page using compareTensors — same code path either way).

export function validateAgainstGPU(
  hookPoint: string,
  atol: number,
  opts: { rtol?: number; gpuDir?: string; cpuDir?: string; promptId?: string } = {},
): CompareResult {
  const gpuDir = opts.gpuDir ?? "dumps/gpu";
  const cpuDir = opts.cpuDir ?? "dumps/cpu";
  const pid = opts.promptId ?? "p0";

  const find = (dir: string): { t: Float32Array; shape: number[] } => {
    const idx = loadIndex(dir);
    const entry = idx.tensors.find((e) => e.hook === hookPoint && e.prompt === pid);
    if (!entry) throw new Error(`${dir}: no dump for ${pid}/${hookPoint}`);
    return { t: loadTensor(dir, entry), shape: squeezeBatch(entry.shape) };
  };

  const gpu = find(gpuDir);
  const cpu = find(cpuDir);
  const causal =
    hookPoint.endsWith("hook_attn_scores") || hookPoint.endsWith("hook_pattern")
      ? { H: cpu.shape[0], T: cpu.shape[1] }
      : undefined;
  return compareTensors(gpu.t, cpu.t, { atol, rtol: opts.rtol, causalLowerOnly: causal });
}

// CI entry: CPU reference vs TransformerLens goldens, key-for-key.
// Tokens come from index.json, so this runs BEFORE the tokenizer exists.

/** Load fp32 weights for the CPU path from convert_weights.py --dtype f32 output. */
export function loadCpuWeights(manifestPath: string, binPath: string): CpuWeights {
  interface Entry { name: string; shape: number[]; dtype: string; byteOffset: number; byteLength: number; }
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as { entries: Entry[] };
  const bin = readFileSync(binPath);
  const w: CpuWeights = {};
  for (const e of manifest.entries) {
    if (e.dtype !== "f32") {
      throw new Error(`CPU validation needs f32 weights (got ${e.dtype} for ${e.name}); ` +
        `run convert_weights.py --dtype f32`);
    }
    w[e.name] = new Float32Array(bin.buffer, bin.byteOffset + e.byteOffset, e.byteLength / 4);
  }
  return w;
}

export interface GoldenReport {
  failures: { prompt: string; hook: string; result: CompareResult }[];
  compared: number;
}

export function validateCpuAgainstGolden(
  fixturesDir: string, cfg: ModelConfig, weights: CpuWeights, atol = 1e-4,
): GoldenReport {
  const idx = loadIndex(fixturesDir);
  const report: GoldenReport = { failures: [], compared: 0 };

  for (const prompt of idx.prompts) {
    const { cache, logits } = forward(prompt.tokens, cfg, weights, { record: "all" });
    for (const entry of idx.tensors.filter((e) => e.prompt === prompt.id)) {
      const golden = loadTensor(fixturesDir, entry);
      const shape = squeezeBatch(entry.shape);
      let ours: Float32Array | undefined;
      if (entry.hook === "logits") ours = logits;
      else ours = cache.get(entry.hook)?.data;
      if (!ours) continue; // e.g. ablation fixtures, handled by a dedicated test
      const causal =
        entry.hook.endsWith("hook_attn_scores") || entry.hook.endsWith("hook_pattern")
          ? { H: shape[0], T: shape[1] }
          : undefined;
      const result = compareTensors(ours, golden, { atol, causalLowerOnly: causal });
      report.compared++;
      if (!result.pass) report.failures.push({ prompt: prompt.id, hook: entry.hook, result });
    }
  }
  return report;
}
