import type { ModelConfig } from "../config.js";
import type { HookName } from "../types.js";

export interface LayerGuess {
  layer: number;            // 0..11, plus 12 = final output for reference
  topTokens: { id: number; token: string; prob: number }[];
}


export class LogitLens {
  constructor(private cfg: ModelConfig, readonly topK = 5) {}

  /** All hooks a logit-lens run must record. */
  hooksToRecord(): HookName[] {
    const hooks: HookName[] = [];
    for (let i = 0; i < this.cfg.n_layers; i++) {
      hooks.push(`blocks.${i}.hook_resid_post` as HookName);
    }
    return hooks;
  }

  
  async compute(
    readback: (hook: HookName) => Promise<Float32Array>,
    decodeToken: (id: number) => string,
    weights: { lnfWeight: Float32Array; lnfBias: Float32Array; wte: Float32Array },
  ): Promise<LayerGuess[]> {
    const { lnfWeight, lnfBias, wte } = weights;
    const D = this.cfg.d_model, V = this.cfg.vocab_size, eps = this.cfg.ln_eps;
    const guesses: LayerGuess[] = [];

    for (let i = 0; i < this.cfg.n_layers; i++) {
      const resid = await readback(`blocks.${i}.hook_resid_post` as HookName); // [T, D]
      const T = resid.length / D;
      const base = (T - 1) * D; // last position's 768-vector

      // final layernorm on the last position (mean/var, then scale+shift)
      let mean = 0;
      for (let d = 0; d < D; d++) mean += resid[base + d];
      mean /= D;
      let variance = 0;
      for (let d = 0; d < D; d++) { const dv = resid[base + d] - mean; variance += dv * dv; }
      variance /= D;
      const rstd = 1 / Math.sqrt(variance + eps);
      const normed = new Float32Array(D);
      for (let d = 0; d < D; d++) normed[d] = (resid[base + d] - mean) * rstd * lnfWeight[d] + lnfBias[d];

      // unembed: logits[v] = normed · wte[v]   (wte is [V, D] row-major, weight tying)
      const logits = new Float32Array(V);
      for (let v = 0; v < V; v++) {
        let acc = 0;
        const wbase = v * D;
        for (let d = 0; d < D; d++) acc += normed[d] * wte[wbase + d];
        logits[v] = acc;
      }

      // softmax normalization (max-subtracted for stability)
      let maxLogit = -Infinity;
      for (let v = 0; v < V; v++) if (logits[v] > maxLogit) maxLogit = logits[v];
      let sumExp = 0;
      for (let v = 0; v < V; v++) sumExp += Math.exp(logits[v] - maxLogit);

      const top = LogitLens.topK(logits, this.topK);
      guesses.push({
        layer: i,
        topTokens: top.map((t) => ({
          id: t.id,
          token: decodeToken(t.id),
          prob: Math.exp(t.logit - maxLogit) / sumExp,
        })),
      });
    }

    return guesses;
  }

  static topK(logits: Float32Array, k: number): { id: number; logit: number }[] {
    const out: { id: number; logit: number }[] = [];
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (out.length < k) {
        out.push({ id: i, logit: v });
        out.sort((a, b) => b.logit - a.logit);
      } else if (v > out[k - 1].logit) {
        out[k - 1] = { id: i, logit: v };
        out.sort((a, b) => b.logit - a.logit);
      }
    }
    return out;
  }
}
