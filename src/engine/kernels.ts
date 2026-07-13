export type KernelName =
  | "embed" | "layernorm" | "matmul_tiled" | "gelu"
  | "softmax_causal" | "residual_add" | "head_zero" | "unembed";

/**
 * KernelRegistry — loads .wgsl source, handles the f16 capability fallback,
 * and caches one GPUComputePipeline per (kernel, variant).
 */

export class KernelRegistry {
  private sources = new Map<KernelName, string>();
  private pipelines = new Map<string, GPUComputePipeline>();
  readonly hasF16: boolean;

  constructor(private device: GPUDevice) {
    this.hasF16 = device.features.has("shader-f16");
  }

  /** Call once at startup. Kernel files are tiny; fetch them all. */
  async loadAll(baseUrl = "/src/kernels"): Promise<void> {
    const names: KernelName[] = [
      "embed", "layernorm", "matmul_tiled", "gelu",
      "softmax_causal", "residual_add", "head_zero", "unembed",
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
    const module = this.device.createShaderModule({ code: this.sourceFor(name), label: name });
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
      label: key,
    });
    this.pipelines.set(key, pipeline);
    return pipeline;
  }


  encodeDispatch(
    encoder: GPUCommandEncoder,
    name: KernelName,
    buffers: GPUBuffer[],
    workgroups: [number, number, number],
  ): void {
    const pipeline = this.getPipeline(name);

    // Connect each buffer to its @binding slot, in order:
    // binding 0 = uniforms (Dims), then inputs, output last.
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, i) => ({ binding: i, resource: { buffer } })),
      label: name,
    });

    const pass = encoder.beginComputePass({ label: name });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
    pass.end();
  }
}
