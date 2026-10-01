/**
 * GpuProfiler - opt-in per-dispatch GPU timing via the `timestamp-query`
 * feature. When attached to a KernelRegistry, every dispatch gets its own
 * compute pass bracketed by begin/end timestamps (standard WebGPU only allows
 * timestamps at pass boundaries). That changes pass structure, so per-kernel
 * numbers come from profiled runs and end-to-end latency from unprofiled ones.
 *
 * Usage per run: reset() → encode dispatches → resolve(encoder) before
 * finish() → submit → await collect().
 */
export interface KernelTiming {
  label: string;
  ms: number;
}

export class GpuProfiler {
  private readonly querySet: GPUQuerySet;
  private readonly resolveBuf: GPUBuffer;
  private readonly readBuf: GPUBuffer;
  private labels: string[] = [];

  constructor(private device: GPUDevice, private readonly capacity = 1024) {
    this.querySet = device.createQuerySet({ type: "timestamp", count: capacity * 2 });
    this.resolveBuf = device.createBuffer({
      size: capacity * 2 * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.readBuf = device.createBuffer({
      size: capacity * 2 * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  static supported(device: GPUDevice): boolean {
    return device.features.has("timestamp-query");
  }

  reset(): void {
    this.labels = [];
  }

  /** Timestamp writes for the next dispatch's pass. */
  next(label: string): GPUComputePassTimestampWrites {
    const i = this.labels.length;
    if (i >= this.capacity) throw new Error(`GpuProfiler capacity ${this.capacity} exceeded`);
    this.labels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 };
  }

  /** Encode the query resolve + copy; call once, after the last dispatch. */
  resolve(encoder: GPUCommandEncoder): void {
    const n = this.labels.length * 2;
    if (n === 0) return;
    encoder.resolveQuerySet(this.querySet, 0, n, this.resolveBuf, 0);
    encoder.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, n * 8);
  }

  /** Per-dispatch GPU durations for the last resolved run, in encode order. */
  async collect(): Promise<KernelTiming[]> {
    const n = this.labels.length * 2;
    if (n === 0) return [];
    await this.readBuf.mapAsync(GPUMapMode.READ, 0, n * 8);
    const ts = new BigUint64Array(this.readBuf.getMappedRange(0, n * 8).slice(0));
    this.readBuf.unmap();
    return this.labels.map((label, i) => ({ label, ms: Number(ts[2 * i + 1] - ts[2 * i]) / 1e6 }));
  }
}
