import type { GpuProfiler } from "./profiler.js";

export type KernelName =
  | "embed" | "layernorm" | "matmul_tiled" | "gelu"
  | "softmax_causal" | "residual_add" | "head_zero" | "unembed"
  | "attn_scores" | "attn_z";

/**
 * KernelRegistry - loads .wgsl source, handles the f16 capability fallback,
 * and caches one GPUComputePipeline per (kernel, variant).
 */

export class KernelRegistry {
  private sources = new Map<KernelName, string>();
  private pipelines = new Map<string, GPUComputePipeline>();
  readonly hasF16: boolean;
  /** When set, every dispatch is timed in its own pass (benchmarking only). */
  profiler: GpuProfiler | null = null;

  constructor(private device: GPUDevice) {
    this.hasF16 = device.features.has("shader-f16");
  }

  /** Call once at startup. Kernel files are tiny; fetch them all. */
  async loadAll(baseUrl = "/src/kernels"): Promise<void> {
    const names: KernelName[] = [
      "embed", "layernorm", "matmul_tiled", "gelu",
      "softmax_causal", "residual_add", "head_zero", "unembed",
      "attn_scores", "attn_z",
    ];
    await Promise.all(names.map(async (n) => {
      const src = await (await fetch(`${baseUrl}/${n}.wgsl`)).text();
      this.sources.set(n, src);
    }));
  }

  /**
   * if shader-f16 is unavailable, strip `enable f16;`
   * and substitute f16 → f32 storage types. One variant, 2× memory, zero
   * divergent code paths.
   */
  private sourceFor(name: KernelName): string {
    const src = this.sources.get(name);
    if (!src) throw new Error(`Kernel source not loaded: ${name}`);
    if (this.hasF16) return src;
    return src.replace(/enable f16;\s*/g, "").replace(/\bf16\b/g, "f32");
  }

  getPipeline(name: KernelName): GPUComputePipeline {
    const key = `${name}:${this.hasF16 ? "f16" : "f32"}`;
    const cached = this.pipelines.get(key);
    if (cached) return cached;
    const module = this.device.createShaderModule({
      code: this.sourceFor(name), 
      label: name 
    });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module: module, entryPoint: "main" },
      label: key,
    });
    this.pipelines.set(key, pipeline);
    return pipeline;
  }


  // Per-dispatch objects, memoized across runs. Uniform contents depend only on
  // T (plus constants), and bind groups on which buffers they bind, so a
  // repeated sequence length re-encodes with zero new GPU objects.
  private uniforms = new Map<string, GPUBuffer>();
  private bindGroups = new Map<string, GPUBindGroup>();
  private bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;

  /** A uniform buffer holding `fields` (u32/f32, padded to 16 bytes), shared by every caller with the same values. */
  uniform(fields: Array<["u32" | "f32", number]>): GPUBuffer {
    const key = fields.map(([t, v]) => `${t}:${v}`).join(",");
    const cached = this.uniforms.get(key);
    if (cached) return cached;
    const size = Math.max(16, Math.ceil((fields.length * 4) / 16) * 16);
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: key });
    const view = new DataView(new ArrayBuffer(size));
    fields.forEach(([type, value], i) => {
      if (type === "u32") view.setUint32(i * 4, value, true);
      else view.setFloat32(i * 4, value, true);
    });
    this.device.queue.writeBuffer(buf, 0, view.buffer);
    this.uniforms.set(key, buf);
    return buf;
  }

  private bufferId(b: GPUBuffer): number {
    let id = this.bufferIds.get(b);
    if (id === undefined) { id = this.nextBufferId++; this.bufferIds.set(b, id); }
    return id;
  }

  /**
   * Drop the memoized uniforms + bind groups once they outgrow `limit`
   * (each distinct T adds ~150 bind groups). Only call between runs: the
   * uniforms are destroyed, so nothing in flight may reference them.
   */
  trimCaches(limit = 2048): void {
    if (this.bindGroups.size + this.uniforms.size <= limit) return;
    for (const b of this.uniforms.values()) b.destroy();
    this.uniforms.clear();
    this.bindGroups.clear();
  }

  /**
   * Encode one dispatch into the recorder's open compute pass.
   * `buffers` bind in order: binding 0 = uniforms (Dims), then inputs, output last.
   */
  encodeDispatch(
    rec: PassRecorder,
    name: KernelName,
    buffers: GPUBuffer[],
    workgroups: [number, number, number],
  ): void {
    const pipeline = this.getPipeline(name);
    const key = `${name}|${buffers.map((b) => this.bufferId(b)).join(",")}`;
    let bindGroup = this.bindGroups.get(key);
    if (!bindGroup) {
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
        label: name,
      });
      this.bindGroups.set(key, bindGroup);
    }
    const pass = rec.pass(name, this.profiler);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
    if (this.profiler) rec.end(); // profiled: one timed pass per dispatch
  }
}

/**
 * PassRecorder - wraps a command encoder and keeps one compute pass open
 * across consecutive dispatches (WebGPU orders dispatches within a pass, so
 * each one sees the previous one's writes). Buffer copies can't live inside a
 * pass, so copy() closes it; the next dispatch opens a fresh one.
 */
export class PassRecorder {
  private open: GPUComputePassEncoder | null = null;

  constructor(readonly encoder: GPUCommandEncoder) {}

  pass(label: string, profiler: GpuProfiler | null): GPUComputePassEncoder {
    if (profiler) {
      this.end();
      this.open = this.encoder.beginComputePass({ label, timestampWrites: profiler.next(label) });
    }
    this.open ??= this.encoder.beginComputePass({ label: "forward" });
    return this.open;
  }

  copy(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void {
    this.end();
    this.encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
  }

  end(): void {
    this.open?.end();
    this.open = null;
  }
}
