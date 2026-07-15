import type { ModelConfig } from "./config.js";
import type { HookName, RunOptions, RunResult } from "./types.js";
import { HookManager } from "./hooks.js";
import { ActivationCache } from "./cache.js";
import { KernelRegistry } from "./kernels.js";
import { WeightStore } from "./weights.js";
import { GPT2Tokenizer } from "./tokenizer.js";

/**
 * ForwardPass — owns the activation buffers and encodes the full GPT-2
 * forward pass as one command submission, firing hooks between dispatches.
 *
 * The numbered sequence in run() is the implementation contract: kernel
 * dispatches and hook fire-points in exactly this order, matching the CPU
 * reference (src/cpu/reference.ts) op-for-op so golden-tensor validation is
 * key-for-key. If you reorder anything here, reorder it there.
 */
export class ForwardPass {
  readonly hooks: HookManager;
  readonly cache: ActivationCache;

  // Persistent activation buffers, allocated once for n_ctx and reused.
  // Shapes annotated for GPT-2 small (T = live sequence length ≤ 1024):
  //   resid      f16 [T, 768]
  //   normed     f16 [T, 768]      (LN output, reused for ln1/ln2/ln_final)
  //   qkv        f16 [T, 2304]     (c_attn output, viewed as q|k|v)
  //   scores     f16 [12, T, T]
  //   pattern    f16 [12, T, T]
  //   z          f16 [T, 12, 64]   (contiguous == [T, 768] view for W_O matmul)
  //   attn_out   f16 [T, 768]
  //   mlp_hidden f16 [T, 3072]
  //   logits     f32 [T, 50257]
  // TODO(impl): allocate in constructor via device.createBuffer (STORAGE |
  //   COPY_SRC | COPY_DST so every buffer can be hooked/patched).
  //   | complexity: low | blocking: none

  constructor(
    private device: GPUDevice,
    private cfg: ModelConfig,
    private weights: WeightStore,
    private kernels: KernelRegistry,
    private tokenizer: GPT2Tokenizer,
  ) {
    this.cache = new ActivationCache(device);
    this.hooks = new HookManager(device, this.cache, cfg, kernels);
  }

  /**
   * One full forward pass over `tokens`, returning fp32 logits [T, vocab].
   *
   * Per-submission sequence (H = hooks.fire):
   *
   *  1. embed.wgsl(ids, wte, wpe) → resid
   *       H("hook_embed"), H("hook_pos_embed")            // SIMPLIFIED: v1 fires
   *       // both on the fused buffer; split-record is a LATER.md item. Golden
   *       // comparison for these two hooks is skipped until then.
   *       H("blocks.0.hook_resid_pre")  ← on resid
   *  2. For i in 0..11:
   *       a. layernorm(resid, ln_1.{g,b}) → normed         H(`blocks.${i}.ln1.hook_normalized`)
   *       b. matmul(normed, c_attn.w [768,2304], +bias) → qkv
   *          view qkv as q,k,v each [T,12,64]
   *            H(hook_q) H(hook_k) H(hook_v)
   *       c. per head h (v1: 12 small matmuls, see ARCH §3):
   *            matmul(q_h [T,64], k_hᵀ [64,T]) → scores_h  // scale NOT applied here
   *       d. H(hook_attn_scores)                            // recorded pre-softmax;
   *          softmax_causal(scores, scale=1/8) → pattern    // scale+mask+softmax in-kernel
   *          H(hook_pattern)                                 // ← writable (pattern patching)
   *       e. per head: matmul(pattern_h [T,T], v_h [T,64]) → z_h
   *          H(hook_z)                                       // ← writable: ablation fires here,
   *                                                          //   AFTER z is produced, BEFORE W_O
   *       f. matmul(z viewed [T,768], c_proj.w, +bias) → attn_out   H(hook_attn_out)
   *       g. residual_add(resid, attn_out) → resid           H(hook_resid_mid)
   *       h. layernorm(resid, ln_2.{g,b}) → normed           H(ln2.hook_normalized)
   *       i. matmul(normed, c_fc.w [768,3072], +bias) → mlp_hidden   H(mlp.hook_pre)
   *       j. gelu(mlp_hidden) → mlp_hidden                   H(mlp.hook_post)
   *       k. matmul(mlp_hidden, c_proj.w [3072,768], +bias) → mlp_out  H(hook_mlp_out)
   *       l. residual_add(resid, mlp_out) → resid            H(hook_resid_post)
   *          // resid now IS blocks.{i+1}.hook_resid_pre — fire that alias too,
   *          // so registrations on either name work (TL treats them as equal values).
   *  3. layernorm(resid, ln_f.{g,b}) → normed                H("ln_final.hook_normalized")
   *  4. unembed(normed, wte transposed-access) → logits (f32, no bias — weight tying)
   *  5. submit; map logits staging buffer; resolve recorded readbacks lazily.
   */
  async run(tokens: Uint32Array, opts: RunOptions = {}): Promise<RunResult> {
    if (tokens.length > this.cfg.n_ctx)
      throw new Error(`Sequence length ${tokens.length} exceeds n_ctx ${this.cfg.n_ctx}`);
    // TODO(impl): register opts.record / opts.writes on this.hooks; create
    //   command encoder; encode the sequence above via kernels.encodeDispatch,
    //   calling hooks.fire(encoder, name, buffer, runId) at each H point;
    //   copy logits to MAP_READ staging; submit; await map; return.
    //   This method is the heart of the project — see ARCHITECTURE.md §2.
    //   | complexity: high | blocking: all 8 kernels, HookManager.fire,
    //   KernelRegistry.encodeDispatch, WeightStore.uploadShard
    void opts;
    throw new Error("ForwardPass.run not implemented");
  }

  /**
   * Greedy decode helper for the UI. NO KV cache in v1 — full recompute per
   * token (ARCHITECTURE.md perf budget clears at 124M). Resist optimizing this
   * before the launch; it is fast enough and the recompute keeps hooks trivially
   * correct for every generated position.
   */
  async generate(prompt: string, maxNewTokens: number, opts: RunOptions = {}): Promise<string> {
    // TODO(impl): encode prompt; loop { run; argmax over logits[T-1]; append;
    //   stop on endoftext (50256) }; decode. Re-fires hooks every step — that is
    //   intentional (interventions must persist across generated tokens).
    //   | complexity: low | blocking: ForwardPass.run
    void prompt; void maxNewTokens; void opts;
    throw new Error("ForwardPass.generate not implemented");
  }
}
