// Takes GPT-2 weights from a CDN and gets them into GPU memory as named buffers for kernels to use.

import type { Dtype } from "./types.js";

/** One entry in manifest.json produced by scripts/convert_weights.py. */
export interface ManifestEntry {
  /** HF parameter name, e.g. "transformer.h.4.attn.c_attn.weight". */
  name: string;
  shape: number[];
  dtype: Dtype;          // "f16" for all weights in v1
  byteOffset: number;    // into the shard
  byteLength: number;
  shard: string;         // relative URL, "weights.bin" here
}

export interface Manifest {
  model: "gpt2";
  entries: ManifestEntry[];
  totalBytes: number;
}

/**
 * WeightStore — manifest + shard fetch (with Cache API persistence) + GPU upload.
 *
 * IMPORTANT (see ARCHITECTURE.md §6): GPT-2's c_attn / c_proj /
 * c_fc weights are HF Conv1D, stored [d_in, d_out] and applied as y = x @ W + b.
 * The manifest preserves that orientation and matmul_tiled consumes it directly.
 * There are NO transposes anywhere in this codebase. 
 */
export class WeightStore {
  private buffers = new Map<string, GPUBuffer>();
  private shapes = new Map<string, number[]>();

  // Private: building a usable store requires async downloading, and constructors
  // can't be async — so creation goes through the static load() factory instead.
  private constructor(private device: GPUDevice) {}

  static async load(
    device: GPUDevice,
    baseUrl: string,
    onProgress?: (loadedBytes: number, totalBytes: number) => void, // for a loading bar
  ): Promise<WeightStore> {
    const store = new WeightStore(device);

    // Download and parse the table of contents.
    const manifest = (await (await fetch(`${baseUrl}/manifest.json`)).json()) as Manifest;
    const shardNames = [...new Set(manifest.entries.map((e) => e.shard))];
    for (const shard of shardNames) {
      const bytes = await store.fetchShardCached(`${baseUrl}/${shard}`, manifest.totalBytes, onProgress);
      store.uploadShard(bytes, manifest.entries.filter((e) => e.shard === shard));
    }
    return store;
  }

  /** Cache API wrapper: hit → return instant; miss → streamed fetch with progress, then cache.put. */
  private async fetchShardCached(
    url: string,
    totalBytes: number,
    onProgress?: (loaded: number, total: number) => void,
  ): Promise<ArrayBuffer> {
    const cache = await caches.open("poke-weights-v1");
    const hit = await cache.match(url);
    if (hit) return hit.arrayBuffer();

    const resp = await fetch(url);
    if (!resp.ok || !resp.body) throw new Error(`Weight fetch failed: ${resp.status} ${url}`);
    const reader = resp.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, totalBytes);
    }
    const bytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { bytes.set(c, off); off += c.byteLength; }
    // clone into cache for next visit; failure to cache is non-fatal (private mode quotas)
    try { await cache.put(url, new Response(bytes.slice().buffer)); } catch { /* ignore */ }
    return bytes.buffer;
  }

  /** Slice shard bytes per manifest entry into individual STORAGE buffers. */
  private uploadShard(shard: ArrayBuffer, entries: ManifestEntry[]): void {
    for (const entry of entries) {
      // WebGPU requires buffer sizes to be a multiple of 4 bytes.
      const size = Math.ceil(entry.byteLength / 4) * 4;

      const buffer = this.device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: entry.name,
      });

      // Copy this weight's slice out of the big blob into its own GPU buffer.
      this.device.queue.writeBuffer(buffer, 0, shard, entry.byteOffset, entry.byteLength);

      this.buffers.set(entry.name, buffer);
      this.shapes.set(entry.name, entry.shape);
    }
  }

  getBuffer(name: string): GPUBuffer {
    const b = this.buffers.get(name);
    if (!b) throw new Error(`Unknown weight: ${name}`);
    return b;
  }

  getShape(name: string): number[] {
    const s = this.shapes.get(name);
    if (!s) throw new Error(`Unknown weight: ${name}`);
    return s;
  }
}
