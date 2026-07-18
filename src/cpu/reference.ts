/**
 * CPU reference implementation — the ORACLE
 *
 * Plain fp32 TypeScript, zero browser APIs, runs in Node. Fully implemented:
 * File's correctness is checked externally (CI diffs it against TransformerLens golden fixtures,
 * key-for-key), not trusted.
 *
 * Op order, epsilon, and gelu constants are the contract the WGSL path must
 * match exactly. If you change anything here, change the kernel and re-run
 * goldens. Every function maps 1:1 to a kernel in src/kernels/.
 *
 * Weight naming: HF GPT-2 names with any "transformer." prefix stripped
 * (convert_weights.py normalizes): wte.weight, wpe.weight,
 * h.{i}.ln_1.{weight,bias}, h.{i}.attn.c_attn.{weight,bias},
 * h.{i}.attn.c_proj.{weight,bias}, h.{i}.ln_2.{weight,bias},
 * h.{i}.mlp.c_fc.{weight,bias}, h.{i}.mlp.c_proj.{weight,bias},
 * ln_f.{weight,bias}.
 *
 * ORIENTATION: all projection weights are HF layout [d_in, d_out],
 * applied y = x @ W + b. No transposes needed.
 */

import {
  GELU_SQRT_2_OVER_PI,
  GELU_TANH_C,
  type ModelConfig, //note interface only exists at compile time
} from "../engine/config.js";

export type CpuWeights = Record<string, Float32Array>;

export interface CpuTensor {
  data: Float32Array;
  shape: number[];
}

/** In-place edit applied at a hook point (CPU analogue of a HookWrite). */
export type CpuHookEdit = (data: Float32Array, shape: number[]) => void;

export interface CpuRunOptions {
  /** Hook names to record. "all" records everything (validation mode). */
  record?: "all" | string[];
  /** Edits applied at hook points, after the tensor is produced and BEFORE
   *  recording or downstream use, reads observe the patched value, matching
   *  HookManager semantics. */
  edits?: Record<string, CpuHookEdit>; 
  // a dictionary where keys are hook names and values are those modifier functions
}

export interface CpuRunResult {
  logits: Float32Array; // [T, vocab]
  cache: Map<string, CpuTensor>;
}


// Per-op functions (1:1 with WGSL kernels)

/** layernorm.wgsl — two-pass mean/variance, BIASED variance, eps inside sqrt. */
export function layerNorm(
  x: Float32Array, rows: number, D: number,
  gamma: Float32Array, beta: Float32Array, eps: number,
): { out: Float32Array; normalized: Float32Array } {
  // `normalized` = (x - mean) / std, BEFORE gamma/beta — this is what
  // TransformerLens records at hook_normalized. `out` = normalized*gamma + beta,
  // the value that feeds the next layer. Hooking the wrong one silently breaks
  // golden validation on every ln*.hook_normalized (see LATER.md).
  const out = new Float32Array(rows * D);
  const normalized = new Float32Array(rows * D);
  for (let r = 0; r < rows; r++) {
    const base = r * D;
    let mean = 0;
    for (let j = 0; j < D; j++) mean += x[base + j];
    mean /= D;
    let varSum = 0;
    for (let j = 0; j < D; j++) {
      const d = x[base + j] - mean;
      varSum += d * d;
    }
    const rstd = 1 / Math.sqrt(varSum / D + eps);
    for (let j = 0; j < D; j++) {
      const n = (x[base + j] - mean) * rstd;
      normalized[base + j] = n;
      out[base + j] = n * gamma[j] + beta[j];
    }
  }
  return { out, normalized };
}

/** gelu.wgsl — gelu_new (tanh approximation). */
export function geluNew(x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const inner = GELU_SQRT_2_OVER_PI * (v + GELU_TANH_C * v * v * v);
    y[i] = 0.5 * v * (1 + Math.tanh(inner));
  }
  return y;
}

/** matmul_tiled.wgsl — y[T,Dout] = x[T,Din] @ W[Din,Dout] + b. HF orientation. 
 * Matmul fron [T, D_in] -> [T, D_out] by [T, D_in] * [D_in, D_out] learned matrix
*/
export function linear(
  x: Float32Array, T: number, Din: number, Dout: number,
  W: Float32Array, b: Float32Array | null,
): Float32Array {
  const y = new Float32Array(T * Dout);
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < Dout; o++) {
      let acc = 0;
      for (let i = 0; i < Din; i++) acc += x[t * Din + i] * W[i * Dout + o];
      y[t * Dout + o] = b ? acc + b[o] : acc;
    }
  }
  return y;
}

/** residual_add.wgsl */
export function residualAdd(a: Float32Array, b: Float32Array): Float32Array {
  const y = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
  return y;
}

/** head_zero.wgsl — z' = z * mask[head] over [T, H, Dh]. */
export function headMask(
  z: Float32Array, T: number, H: number, Dh: number, mask: Float32Array,
): void {
  for (let i = 0; i < T * H * Dh; i++) {
    z[i] *= mask[((i / Dh) | 0) % H];
  }
}

// Forward pass

const MASK_FILL = -1e9; // recorded value at masked score positions (display only;
                        // softmax excludes them from max/sum and writes exact 0)

export function forward(
  tokens: ArrayLike<number>, cfg: ModelConfig, w: CpuWeights,
  opts: CpuRunOptions = {},
): CpuRunResult {
  const T = tokens.length;
  const { d_model: D, n_heads: H, d_head: Dh, d_ff: F, vocab_size: V } = cfg;
  if (T > cfg.n_ctx) throw new Error(`seq len ${T} > n_ctx ${cfg.n_ctx}`);

  const cache = new Map<string, CpuTensor>();
  const wantAll = opts.record === "all";
  const wanted = new Set(wantAll ? [] : (opts.record ?? []));
  const hook = (name: string, data: Float32Array, shape: number[]): void => {
    opts.edits?.[name]?.(data, shape); //call a function if one is registered in opts else skip it
    if (wantAll || wanted.has(name)) {
      cache.set(name, { data: data.slice(), shape: [...shape] }); //make a copy of numbers and shape if caller asked to record this hook
    }
  };
  
  //weight lookup and return function
  const req = (name: string): Float32Array => {
    const t = w[name];
    if (!t) throw new Error(`missing weight: ${name}`);
    return t;
  };

  // 1. Embedding (CPU records token/positional parts separately; the fused
  //    engine kernel fires them on the summed buffer its simplified there).
  const wte = req("wte.weight"), wpe = req("wpe.weight"); //fetches embedding and positial matrices
  const tokEmb = new Float32Array(T * D);
  const posEmb = new Float32Array(T * D);
  let resid: Float32Array = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const id = tokens[t];
    if (id < 0 || id >= V) throw new Error(`token id ${id} out of range`);
    for (let d = 0; d < D; d++) {
      tokEmb[t * D + d] = wte[id * D + d];
      posEmb[t * D + d] = wpe[t * D + d];
      resid[t * D + d] = tokEmb[t * D + d] + posEmb[t * D + d];
    }
  }
  hook("hook_embed", tokEmb, [T, D]);
  hook("hook_pos_embed", posEmb, [T, D]);

  // 2. Blocks
  for (let i = 0; i < cfg.n_layers; i++) {
    const p = `blocks.${i}`;
    hook(`${p}.hook_resid_pre`, resid, [T, D]);

    // a. LN1  — hook the pre-γβ `normalized`, use post-γβ `out` for compute
    const ln1 = layerNorm(resid, T, D, req(`h.${i}.ln_1.weight`), req(`h.${i}.ln_1.bias`), cfg.ln_eps);
    hook(`${p}.ln1.hook_normalized`, ln1.normalized, [T, D]);
    let normed = ln1.out;

    // b. QKV projection [T, 3D], split into [T, 12, 64] views
    const qkv = linear(normed, T, D, 3 * D, req(`h.${i}.attn.c_attn.weight`), req(`h.${i}.attn.c_attn.bias`));
    const q = new Float32Array(T * H * Dh), k = new Float32Array(T * H * Dh), v = new Float32Array(T * H * Dh);
    for (let t = 0; t < T; t++)
      for (let h = 0; h < H; h++)
        for (let d = 0; d < Dh; d++) {
          const dst = (t * H + h) * Dh + d;
          q[dst] = qkv[t * 3 * D + h * Dh + d];
          k[dst] = qkv[t * 3 * D + D + h * Dh + d];
          v[dst] = qkv[t * 3 * D + 2 * D + h * Dh + d];
        }
    hook(`${p}.attn.hook_q`, q, [T, H, Dh]);
    hook(`${p}.attn.hook_k`, k, [T, H, Dh]);
    hook(`${p}.attn.hook_v`, v, [T, H, Dh]);

    // c+d. Scaled scores (scale folded HERE, matching softmax_causal.wgsl),
    //      causal mask, softmax with valid-max subtraction, masked → exact 0.
    const scale = 1 / Math.sqrt(Dh); //scaling factor prevents dot products from getting too large pre SM
    const scores = new Float32Array(H * T * T);
    for (let h = 0; h < H; h++)
      for (let qi = 0; qi < T; qi++)
        for (let ki = 0; ki < T; ki++) {
          const idx = (h * T + qi) * T + ki;
          if (ki > qi) { scores[idx] = MASK_FILL; continue; } //casual mask
          let acc = 0;
          for (let d = 0; d < Dh; d++)
            acc += q[(qi * H + h) * Dh + d] * k[(ki * H + h) * Dh + d];
          scores[idx] = acc * scale;
        }
    hook(`${p}.attn.hook_attn_scores`, scores, [H, T, T]);

    //softmax the scores
    const pattern = new Float32Array(H * T * T); // masked positions (future tokens) will stay exactly zero, never touched.

    for (let h = 0; h < H; h++)
      for (let qi = 0; qi < T; qi++) {
        const base = (h * T + qi) * T;
        let max = -Infinity;
        for (let ki = 0; ki <= qi; ki++) max = Math.max(max, scores[base + ki]);
        let sum = 0;
        for (let ki = 0; ki <= qi; ki++) {
          const e = Math.exp(scores[base + ki] - max);
          pattern[base + ki] = e;
          sum += e;
        }
        for (let ki = 0; ki <= qi; ki++) pattern[base + ki] /= sum;
        // ki > qi positions stay exact 0
      }
    hook(`${p}.attn.hook_pattern`, pattern, [H, T, T]);

    // e. z = pattern @ v, per head → [T, H, Dh]
    const z = new Float32Array(T * H * Dh);
    for (let h = 0; h < H; h++)
      for (let qi = 0; qi < T; qi++)
        for (let d = 0; d < Dh; d++) {
          let acc = 0;
          for (let ki = 0; ki <= qi; ki++)
            acc += pattern[(h * T + qi) * T + ki] * v[(ki * H + h) * Dh + d];
          z[(qi * H + h) * Dh + d] = acc;
        }
    hook(`${p}.attn.hook_z`, z, [T, H, Dh]); // ← ablation edit fires here

    // f. Output projection: z viewed [T, D] (contiguous because layout is [T,H,Dh])
    const attnOut = linear(z, T, D, D, req(`h.${i}.attn.c_proj.weight`), req(`h.${i}.attn.c_proj.bias`));
    hook(`${p}.hook_attn_out`, attnOut, [T, D]);

    // g.
    resid = residualAdd(resid, attnOut);
    hook(`${p}.hook_resid_mid`, resid, [T, D]);

    // h. LN2  — hook pre-γβ, compute with post-γβ
    const ln2 = layerNorm(resid, T, D, req(`h.${i}.ln_2.weight`), req(`h.${i}.ln_2.bias`), cfg.ln_eps);
    hook(`${p}.ln2.hook_normalized`, ln2.normalized, [T, D]);
    normed = ln2.out;

    // i–k. MLP
    const pre = linear(normed, T, D, F, req(`h.${i}.mlp.c_fc.weight`), req(`h.${i}.mlp.c_fc.bias`));
    hook(`${p}.mlp.hook_pre`, pre, [T, F]);
    const post = geluNew(pre);
    hook(`${p}.mlp.hook_post`, post, [T, F]);
    const mlpOut = linear(post, T, F, D, req(`h.${i}.mlp.c_proj.weight`), req(`h.${i}.mlp.c_proj.bias`));
    hook(`${p}.hook_mlp_out`, mlpOut, [T, D]);

    // l.
    resid = residualAdd(resid, mlpOut);
    hook(`${p}.hook_resid_post`, resid, [T, D]);
  }

  // 3. Final LN  — hook pre-γβ, unembed the post-γβ output
  const lnF = layerNorm(resid, T, D, req("ln_f.weight"), req("ln_f.bias"), cfg.ln_eps);
  hook("ln_final.hook_normalized", lnF.normalized, [T, D]);
  const normedF = lnF.out;

  // 4. Unembed: logits = normed @ wte^T, NO bias (weight tying). f32 stays f32.
  const logits = new Float32Array(T * V);
  for (let t = 0; t < T; t++)
    for (let vi = 0; vi < V; vi++) {
      let acc = 0;
      for (let d = 0; d < D; d++) acc += normedF[t * D + d] * wte[vi * D + d];
      logits[t * V + vi] = acc;
    }

  return { logits, cache };
}

// Test utilities (used by reference.test.ts)

/** Deterministic PRNG (mulberry32) so tests are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random weights for an arbitrary (micro) config for invariant tests only. */
export function randomWeights(cfg: ModelConfig, seed = 42): CpuWeights {
  const r = rng(seed);
  const t = (n: number, scale = 0.06): Float32Array => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (r() * 2 - 1) * scale;
    return a;
  };
  const ones = (n: number): Float32Array => new Float32Array(n).fill(1);
  const { d_model: D, d_ff: F, vocab_size: V, n_ctx: C } = cfg;
  const w: CpuWeights = {
    "wte.weight": t(V * D, 0.1),
    "wpe.weight": t(C * D, 0.02),
    "ln_f.weight": ones(D),
    "ln_f.bias": t(D, 0.01),
  };
  for (let i = 0; i < cfg.n_layers; i++) {
    w[`h.${i}.ln_1.weight`] = ones(D);
    w[`h.${i}.ln_1.bias`] = t(D, 0.01);
    w[`h.${i}.attn.c_attn.weight`] = t(D * 3 * D);
    w[`h.${i}.attn.c_attn.bias`] = t(3 * D, 0.01);
    w[`h.${i}.attn.c_proj.weight`] = t(D * D);
    w[`h.${i}.attn.c_proj.bias`] = t(D, 0.01);
    w[`h.${i}.ln_2.weight`] = ones(D);
    w[`h.${i}.ln_2.bias`] = t(D, 0.01);
    w[`h.${i}.mlp.c_fc.weight`] = t(D * F);
    w[`h.${i}.mlp.c_fc.bias`] = t(F, 0.01);
    w[`h.${i}.mlp.c_proj.weight`] = t(F * D);
    w[`h.${i}.mlp.c_proj.bias`] = t(D, 0.01);
  }
  return w;
}
