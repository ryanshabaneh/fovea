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

  // Persistent activation buffers, allocated once and reused every run.
  private readonly resid: GPUBuffer;      // [T, 768]   the residual stream
  private readonly normed: GPUBuffer;     // [T, 768]   layernorm output
  private readonly qkv: GPUBuffer;        // [T, 2304]  c_attn output (Q|K|V)
  private readonly scores: GPUBuffer;     // [12, T, T] attention scores
  private readonly pattern: GPUBuffer;    // [12, T, T] post-softmax attention
  private readonly z: GPUBuffer;          // [T, 768]   per-head attn output
  private readonly attnOut: GPUBuffer;    // [T, 768]   after W_O
  private readonly mlpHidden: GPUBuffer;  // [T, 3072]  MLP hidden
  private readonly logits: GPUBuffer;     // [T, 50257] f32 output

  constructor(
    //passed in from outside
    private device: GPUDevice,
    private cfg: ModelConfig,
    private weights: WeightStore,
    private kernels: KernelRegistry,
    private tokenizer: GPT2Tokenizer,
  ) {
    this.cache = new ActivationCache(device);
    this.hooks = new HookManager(device, this.cache, cfg, kernels);
    // Allocate each activation buffer once, sized for the max sequence (n_ctx),
    // and reuse them every run. STORAGE = kernels read/write; COPY_SRC = can be
    // hooked/read back; COPY_DST = can be patched.
    const N = cfg.n_ctx, D = cfg.d_model, F = cfg.d_ff, H = cfg.n_heads, V = cfg.vocab_size;
    const F16 = 2, F32 = 4; // bytes per element
    const mk = (bytes: number) =>
      device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

    this.resid     = mk(N * D * F16);       // [N, 768]
    this.normed    = mk(N * D * F16);       // [N, 768]
    this.qkv       = mk(N * 3 * D * F16);   // [N, 2304]  (3×768)
    this.scores    = mk(H * N * N * F16);   // [12, N, N]
    this.pattern   = mk(H * N * N * F16);   // [12, N, N]
    this.z         = mk(N * D * F16);       // [N, 768]
    this.attnOut   = mk(N * D * F16);       // [N, 768]
    this.mlpHidden = mk(N * F * F16);       // [N, 3072]
    this.logits    = mk(N * V * F32);       // [N, 50257] f32
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

    const T = tokens.length;
    const runId = opts.runId ?? "default";

    this.hooks.clear();

    //build the maps in hooks
    for (const hook of opts.record ?? []) this.hooks.registerRead(hook);
    for (const write of opts.writes ?? []) this.hooks.registerWrite(write);

    const encoder = this.device.createCommandEncoder({label: "forward"});

    // Embed: token ids → resid = wte[id] + wpe[pos]
    const idsBuf = this.device.createBuffer({
      size: Math.ceil((T * 4) / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(idsBuf, 0, new Uint32Array(tokens));

    const embedDims = this.uniform([["u32", T], ["u32", this.cfg.d_model], ["u32", this.cfg.vocab_size]]);
    this.kernels.encodeDispatch(
      encoder,
      "embed",
      [embedDims, idsBuf, this.weights.getBuffer("wte.weight"), this.weights.getBuffer("wpe.weight"), this.resid],
      [Math.ceil((T * this.cfg.d_model) / 256), 1, 1],
    );
    this.hooks.fire(encoder, "blocks.0.hook_resid_pre", this.resid, runId, T);

    const D = this.cfg.d_model;
    const H = this.cfg.n_heads;
    const Dh = this.cfg.d_head;
    const attnScale = 1 / Math.sqrt(Dh); // 1/√64 = 0.125

    for (let i = 0; i < this.cfg.n_layers; i++) {
      const p = `blocks.${i}` as `blocks.${number}`;

      // (a) ln1: normalize resid → normed
      const ln1Dims = this.uniform([["u32", T], ["u32", D], ["f32", this.cfg.ln_eps]]);
      this.kernels.encodeDispatch(
        encoder,
        "layernorm",
        [ln1Dims, this.resid,
         this.weights.getBuffer(`h.${i}.ln_1.weight`),
         this.weights.getBuffer(`h.${i}.ln_1.bias`),
         this.normed],
        [T, 1, 1],
      );
      this.hooks.fire(encoder, `${p}.ln1.hook_normalized`, this.normed, runId, T);

      // (b) W_qkv: normed → qkv = normed @ c_attn.weight + bias   (Q|K|V fused)
      const qkvDims = this.uniform([["u32", T], ["u32", 3 * D], ["u32", D], ["u32", 1]]);
      this.kernels.encodeDispatch(
        encoder,
        "matmul_tiled",
        [qkvDims, this.normed,
         this.weights.getBuffer(`h.${i}.attn.c_attn.weight`),
         this.weights.getBuffer(`h.${i}.attn.c_attn.bias`),
         this.qkv],
        [Math.ceil((3 * D) / 16), Math.ceil(T / 16), 1],
      );

      // (c) attn scores: Q·Kᵀ · scale (all heads) → scores [H,T,T]
      const scoresDims = this.uniform([["u32", T], ["u32", H], ["u32", Dh], ["f32", attnScale]]);
      this.kernels.encodeDispatch(
        encoder,
        "attn_scores",
        [scoresDims, this.qkv, this.scores],
        [Math.ceil((H * T * T) / 256), 1, 1],
      );
      this.hooks.fire(encoder, `${p}.attn.hook_attn_scores`, this.scores, runId, T);

      // (d) softmax over each row (causal). Scale already applied in (c), so pass 1.
      const softDims = this.uniform([["u32", H], ["u32", T], ["f32", 1]]);
      this.kernels.encodeDispatch(
        encoder,
        "softmax_causal",
        [softDims, this.scores, this.pattern],
        [H * T, 1, 1],
      );
      this.hooks.fire(encoder, `${p}.attn.hook_pattern`, this.pattern, runId, T);

      // (e) attn_z: pattern·V (all heads) → z [T,768]   ← ablation fires here
      const zDims = this.uniform([["u32", T], ["u32", H], ["u32", Dh]]);
      this.kernels.encodeDispatch(
        encoder,
        "attn_z",
        [zDims, this.qkv, this.pattern, this.z],
        [Math.ceil((T * D) / 256), 1, 1],
      );
      this.hooks.fire(encoder, `${p}.attn.hook_z`, this.z, runId, T);
    }

    throw new Error("ForwardPass.run: qkv done, attention core next");
  }

  /** Build a uniform buffer from mixed u32/f32 fields (padded to 16 bytes). */
  private uniform(fields: Array<["u32" | "f32", number]>): GPUBuffer {
    const size = Math.max(16, Math.ceil((fields.length * 4) / 16) * 16);
    const buf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const data = new ArrayBuffer(size);
    const view = new DataView(data);
    fields.forEach(([type, value], i) => {
      if (type === "u32") view.setUint32(i * 4, value, true);
      else view.setFloat32(i * 4, value, true);
    });
    this.device.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  /**
   * Greedy decode helper for the UI. full recompute per
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
    throw new Error("ForwardPass.generate not done");
  }
}
