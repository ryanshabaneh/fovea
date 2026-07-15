import type { ModelConfig } from "./config.js";
import type { HookName, HookWrite, TensorMeta } from "./types.js";
import { isWritableHook } from "./types.js";
import type { ActivationCache } from "./cache.js";
import type { KernelRegistry } from "./kernels.js";

/**
 * ForwardPass calls fire() between kernel dispatches with the live GPUBuffer
 * for that hook point. HookManager then either:
 * (a) copies the buffer into the ActivationCache if a read is registered,
 * and (b) mutates the live buffer (head_mask dispatch or buffer-to-buffer copy)
 * if a write is registered.
 * Order within fire(): reads-before-write capture the CLEAN value only when
 * registered with {preWrite: true}; default reads observe the patched value,
 * matching TransformerLens semantics
 */
export class HookManager {
  private reads = new Map<HookName, { preWrite: boolean }>();
  private writes = new Map<HookName, HookWrite>();

  constructor(
    private device: GPUDevice,
    private cache: ActivationCache,
    private cfg: ModelConfig,
    private kernels: KernelRegistry,
  ) {}

  registerRead(hook: HookName, opts: { preWrite?: boolean } = {}): void {
    this.reads.set(hook, { preWrite: opts.preWrite ?? false });
  }

  registerWrite(write: HookWrite): void {
    if (!isWritableHook(write.hook)) {
      throw new Error(`Hook ${write.hook} is not in the write set; see (ARCHITECTURE.md)`);
    }
    this.writes.set(write.hook, write);
  }

  clear(): void {
    this.reads.clear();
    this.writes.clear();
  }

  /**
   * Called by ForwardPass at each hook point, inside command encoding.
   * @param encoder  the active command encoder (copies/dispatches are encoded, not submitted)
   * @param hook     hook name being fired
   * @param buffer   live GPU buffer holding this activation (f16)
   * @param runId    cache key for reads
   */
  fire(encoder: GPUCommandEncoder, hook: HookName, buffer: GPUBuffer, runId: string, seqLen: number): void {
    const read = this.reads.get(hook);
    const write = this.writes.get(hook);
    if (!read && !write) return; // nothing registered here — fast exit

    // Shape and byte size of this activation (f16 = 2 bytes per element).
    const shape = HookManager.hookShape(hook, this.cfg, seqLen);
    const byteLength = shape.reduce((a, b) => a * b, 1) * 2;

    // Records a copy of the live buffer into the cache slot for this run+hook.
    const recordRead = () => {
      const slot = this.cache.ensureSlot(runId, hook, byteLength, shape);
      encoder.copyBufferToBuffer(buffer, 0, slot, 0, byteLength);
    };

    // (1) pre-write read → captures the CLEAN value, before any edit.
    if (read?.preWrite) recordRead();

    // (2) write → edits the live buffer in place.
    if (write) {
      if (write.op.kind === "copy_from") {
        // patching: overwrite this activation with another run's cached buffer
        encoder.copyBufferToBuffer(write.op.src, 0, buffer, 0, write.op.bytes);
      } else {
        // ablation: scale each head by mask via the head_zero kernel
        this.encodeHeadMask(encoder, buffer, write.op.mask, seqLen);
      }
    }

    // (3) post-write read (default) → captures the PATCHED value, after the edit.
    if (read && !read.preWrite) recordRead();
  }

  /**
   * Encode a head_zero dispatch that scales each head of `buffer` (shape
   * [T, H, Dh]) by mask[h]. head_zero writes to a separate output, so we run it
   * into a temp buffer and copy back, leaving the ablated result in `buffer`.
   */
  private encodeHeadMask(
    encoder: GPUCommandEncoder,
    buffer: GPUBuffer,
    mask: Float32Array,
    seqLen: number,
  ): void {
    const T = seqLen, H = this.cfg.n_heads, Dh = this.cfg.d_head;
    const n = T * H * Dh;
    const byteLength = n * 2; // f16

    // Dims uniform {T, H, Dh}. Uniform structs round up to 16 bytes.
    const dims = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(dims, 0, new Uint32Array([T, H, Dh]));

    // Per-head mask, f32 [H].
    const maskBuf = this.device.createBuffer({
      size: Math.ceil((H * 4) / 4) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(maskBuf, 0, new Float32Array(mask));

    // head_zero writes to a separate output; use a temp then copy back into buffer.
    const tmp = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    this.kernels.encodeDispatch(
      encoder,
      "head_zero",
      [dims, buffer, maskBuf, tmp],
      [Math.ceil(n / 256), 1, 1],
    );
    encoder.copyBufferToBuffer(tmp, 0, buffer, 0, byteLength);
  }

  /**
   * Shape of the tensor at a hook point for sequence length T.
   */
  static hookShape(hook: HookName, cfg: ModelConfig, T: number): number[] {
    if (hook.endsWith("hook_attn_scores") || hook.endsWith("hook_pattern"))
      return [cfg.n_heads, T, T];
    if (hook.endsWith("hook_q") || hook.endsWith("hook_k") ||
        hook.endsWith("hook_v") || hook.endsWith("hook_z"))
      return [T, cfg.n_heads, cfg.d_head];
    if (hook.endsWith("mlp.hook_pre") || hook.endsWith("mlp.hook_post"))
      return [T, cfg.d_ff];
    // all residual-stream-width hooks
    return [T, cfg.d_model];
  }

  static hookMeta(hook: HookName, cfg: ModelConfig, T: number): TensorMeta {
    return { name: hook, shape: HookManager.hookShape(hook, cfg, T), dtype: "f16" };
  }
}
