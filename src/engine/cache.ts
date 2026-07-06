import type { HookName } from "./types.js";

interface Slot {
  buffer: GPUBuffer;     // f16, COPY_DST | COPY_SRC (so it can also be a patch source)
  byteLength: number;
  shape: number[];
}

/**
 * Decode one IEEE-754 half-float (given as a 16-bit int) to f32.
 * Handles subnormals, inf, and NaN. 
 */
function halfToFloat(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  const s = sign ? -1 : 1;
  if (exp === 0) {
    // subnormal (or zero): no implicit leading 1
    return s * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    // inf / nan
    return frac ? NaN : s * Infinity;
  }
  // normal: implicit leading 1
  return s * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/** Precomputed f16-bits → f32 table. 65536 entries, built once at module load. */
const F16_TO_F32 = (() => {
  const lut = new Float32Array(65536);
  for (let i = 0; i < 65536; i++) lut[i] = halfToFloat(i);
  return lut;
})();

/**
 * ActivationCache, owns GPU-side copies of recorded activations
 */

export class ActivationCache {
  private slots = new Map<string, Slot>();

  constructor(private device: GPUDevice) {}

  private key(runId: string, hook: HookName): string {
    return `${runId}::${hook}`;
  }

  /** Create (or reuse) the destination buffer a read-hook copies into. */
  ensureSlot(runId: string, hook: HookName, byteLength: number, shape: number[]): GPUBuffer {
    const k = this.key(runId, hook);
    const existing = this.slots.get(k);
    if (existing && existing.byteLength === byteLength) return existing.buffer;
    existing?.buffer.destroy(); //executes if existing is undefined
    const buffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: k,
    });
    this.slots.set(k, { buffer, byteLength, shape });
    return buffer;
  }

  getBuffer(runId: string, hook: HookName): GPUBuffer {
    const slot = this.slots.get(this.key(runId, hook));
    if (!slot) throw new Error(`No cached activation for ${runId}::${hook}`);
    return slot.buffer;
  }

  getShape(runId: string, hook: HookName): number[] {
    const slot = this.slots.get(this.key(runId, hook));
    if (!slot) throw new Error(`No cached activation for ${runId}::${hook}`);
    return slot.shape;
  }

  /**
   * Read an activation back to the CPU as Float32Array (f16 → f32 conversion
   * happens here, once, at the edge never mid-forward).
   */
  async readback(runId: string, hook: HookName): Promise<Float32Array> {
    const slot = this.slots.get(this.key(runId, hook));
    if (!slot) throw new Error(`No cached activation for ${runId}::${hook}`);

    // 1. staging buffer the CPU can map
    const staging = this.device.createBuffer({
      size: slot.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: `readback ${runId}::${hook}`,
    });

    // 2. copy GPU activation → staging, submit
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(slot.buffer, 0, staging, 0, slot.byteLength);
    this.device.queue.submit([encoder.finish()]);

    // 3. wait for the copy, then map into CPU memory
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(staging.getMappedRange());

    // 4. decode each f16 → f32 via the Look up table (LUT)
    const out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = F16_TO_F32[raw[i]];

    staging.unmap();
    staging.destroy();
    return out;
  }

  destroy(): void {
    for (const s of this.slots.values()) s.buffer.destroy();
    this.slots.clear();
  }
}
