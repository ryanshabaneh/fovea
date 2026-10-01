/**
 * Benchmark page (bench.html, dev server only - not part of the app build).
 *
 * For each sequence length T:
 *   - end-to-end: wall-clock fwd.run() (encode → submit → logits readback),
 *     unprofiled, median of N runs after warmup
 *   - per-kernel GPU ms: profiled runs (timestamp-query, one pass per
 *     dispatch), summed per kernel within a run, median across runs
 *   - logit hashes for a plain run, an ablated run, and a pattern readback,
 *     so configurations can be checked for bit-identical numerics
 * Plus a matmul microbenchmark at the model's real shapes.
 *
 * Results render on the page and are POSTed to /__bench when the runner
 * (scripts/bench.mjs) is serving it.
 */
import { boot } from "../engine/boot.js";
import { GpuProfiler } from "../engine/profiler.js";
import { HeadAblation } from "../engine/interventions/ablation.js";
import type { HookName } from "../engine/types.js";
import tiledSrc from "../kernels/matmul_tiled.wgsl?raw";
import unembedSrc from "../kernels/unembed.wgsl?raw";
import naiveSrc from "./kernels/matmul_naive.wgsl?raw";
import unembedStridedSrc from "./kernels/unembed_strided.wgsl?raw";

const params = new URLSearchParams(location.search);
const LABEL = params.get("label") ?? "unlabeled";
const TS = (params.get("T") ?? "8,64,256,1024").split(",").map(Number);
const out = document.getElementById("out")!;
const log = (s: string): void => { out.textContent += s + "\n"; console.log(s); };

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** FNV-1a over the raw bytes - bit-identical outputs give identical hashes. */
function hash(arr: ArrayBufferView): string {
  const u8 = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let h = 0x811c9dc5;
  for (let i = 0; i < u8.length; i++) { h ^= u8[i]; h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16).padStart(8, "0");
}

/** Deterministic token ids (latency doesn't depend on which tokens). */
function tokensFor(T: number): Uint32Array {
  const ids = new Uint32Array(T);
  let s = 12345;
  for (let i = 0; i < T; i++) { s = (Math.imul(s, 1103515245) + 12345) >>> 0; ids[i] = s % 50256; }
  return ids;
}

const nextFrame = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

async function main(): Promise<void> {
  log(`label=${LABEL} T=${TS.join(",")}`);
  const { device, fwd, kernels, hasF16 } = await boot({ features: ["timestamp-query"] });
  if (!hasF16) throw new Error("bench expects shader-f16");
  if (!GpuProfiler.supported(device)) throw new Error("timestamp-query unavailable");
  const adapterInfo = (device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo;
  const prof = new GpuProfiler(device);

  const model: Record<number, unknown> = {};
  for (const T of TS) {
    const tokens = tokensFor(T);
    const N = T >= 1024 ? 10 : 20;
    kernels.profiler = null;
    for (let i = 0; i < 3; i++) await fwd.run(tokens); // warmup: pipelines, plans, caches

    const e2e: number[] = [];
    let logitsHash = "";
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      const { logits } = await fwd.run(tokens);
      e2e.push(performance.now() - t0);
      if (i === 0) logitsHash = hash(logits);
      await nextFrame();
    }

    // The app's path: one head ablated (+ pattern readback when a head is clicked).
    const ablate = new HeadAblation(5, 1).toHookWrite();
    const patHook = "blocks.5.attn.hook_pattern" as HookName;
    const ab: number[] = [];
    let ablatedHash = "", patternHash = "";
    for (let i = 0; i < N; i++) {
      const t0 = performance.now();
      const r = await fwd.run(tokens, { writes: [ablate] });
      ab.push(performance.now() - t0);
      if (i === 0) ablatedHash = hash(r.logits);
    }
    const rp = await fwd.run(tokens, { writes: [ablate], record: [patHook], runId: "bench" });
    patternHash = hash(rp.logits) + ":" + hash(await fwd.cache.readback(rp.runId, patHook));

    // Profiled runs: per-kernel GPU time.
    kernels.profiler = prof;
    const perKernelRuns: Record<string, number[]> = {};
    const gpuTotals: number[] = [];
    let dispatches = 0;
    for (let i = 0; i < 5; i++) {
      prof.reset();
      await fwd.run(tokens);
      const timings = await prof.collect();
      dispatches = timings.length;
      const sums: Record<string, number> = {};
      for (const t of timings) sums[t.label] = (sums[t.label] ?? 0) + t.ms;
      for (const [k, v] of Object.entries(sums)) (perKernelRuns[k] ??= []).push(v);
      gpuTotals.push(timings.reduce((a, t) => a + t.ms, 0));
    }
    kernels.profiler = null;
    const perKernel = Object.fromEntries(Object.entries(perKernelRuns).map(([k, v]) => [k, median(v)]));

    model[T] = {
      e2eMs: { median: median(e2e), min: Math.min(...e2e), n: N },
      ablatedE2eMs: { median: median(ab), min: Math.min(...ab), n: N },
      gpuMs: median(gpuTotals),
      dispatches,
      perKernelGpuMs: perKernel,
      hashes: { logits: logitsHash, ablated: ablatedHash, pattern: patternHash },
    };
    log(`T=${T} e2e ${median(e2e).toFixed(2)} ms · ablated ${median(ab).toFixed(2)} ms · gpu ${median(gpuTotals).toFixed(2)} ms · ${dispatches} dispatches · logits ${logitsHash}`);
  }

  const micro = await matmulMicro(device, prof);
  const result = { label: LABEL, date: new Date().toISOString(), userAgent: navigator.userAgent, adapter: adapterInfo ? { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture } : null, model, micro };
  log(JSON.stringify(result, null, 2));
  try { await fetch("/__bench", { method: "POST", body: JSON.stringify(result) }); } catch { /* page opened by hand */ }
  log("done");
}

// ── matmul microbenchmark ────────────────────────────────────────────────────

function randomF16(n: number): Uint16Array {
  // small normal halves in roughly ±[2^-5, 1): sign | exp 10..14 | mantissa
  const a = new Uint16Array(n);
  let s = 987654321;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    a[i] = ((s >>> 31) << 15) | ((10 + ((s >>> 16) % 5)) << 10) | (s & 0x3ff);
  }
  return a;
}

async function matmulMicro(device: GPUDevice, prof: GpuProfiler): Promise<unknown> {
  const pipe = (code: string, label: string): GPUComputePipeline =>
    device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code, label }), entryPoint: "main" }, label });
  const tiled = pipe(tiledSrc, "matmul_tiled");
  const naive = pipe(naiveSrc, "matmul_naive");
  const unembedNew = pipe(unembedSrc, "unembed");
  const unembedOld = pipe(unembedStridedSrc, "unembed_strided");

  const storage = (data: ArrayBufferView | number): GPUBuffer => {
    const size = Math.ceil((typeof data === "number" ? data : data.byteLength) / 4) * 4;
    const b = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (typeof data !== "number") device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  };

  async function readHash(buf: GPUBuffer, bytes: number): Promise<string> {
    const st = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, bytes);
    device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const h = hash(new Uint8Array(st.getMappedRange()));
    st.unmap(); st.destroy();
    return h;
  }

  /** Median GPU ms per dispatch over R timed dispatches (after 3 warmups). */
  async function time(p: GPUComputePipeline, bufs: GPUBuffer[], grid: [number, number]): Promise<number> {
    const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: bufs.map((buffer, i) => ({ binding: i, resource: { buffer } })) });
    const R = 20;
    prof.reset();
    const enc = device.createCommandEncoder();
    for (let i = 0; i < R + 3; i++) {
      const pass = enc.beginComputePass({ timestampWrites: prof.next("m") });
      pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(grid[0], grid[1]);
      pass.end();
    }
    prof.resolve(enc);
    device.queue.submit([enc.finish()]);
    return median((await prof.collect()).slice(3).map((t) => t.ms));
  }

  const shapes = [
    { name: "c_attn 768→2304", K: 768, N: 2304 },
    { name: "c_proj 768→768", K: 768, N: 768 },
    { name: "c_fc 768→3072", K: 768, N: 3072 },
    { name: "mlp c_proj 3072→768", K: 3072, N: 768 },
  ];
  const rows: unknown[] = [];
  for (const T of TS) {
    for (const s of shapes) {
      const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(dims, 0, new Uint32Array([T, s.N, s.K, 1]));
      const A = storage(randomF16(T * s.K)), B = storage(randomF16(s.K * s.N)), bias = storage(randomF16(s.N));
      const C1 = storage(T * s.N * 2), C2 = storage(T * s.N * 2);
      const grid: [number, number] = [Math.ceil(s.N / 16), Math.ceil(T / 16)];
      const naiveMs = await time(naive, [dims, A, B, bias, C1], grid);
      const tiledMs = await time(tiled, [dims, A, B, bias, C2], grid);
      const same = (await readHash(C1, T * s.N * 2)) === (await readHash(C2, T * s.N * 2));
      rows.push({ T, shape: s.name, naiveMs, tiledMs, speedup: naiveMs / tiledMs, bitIdentical: same });
      log(`micro T=${T} ${s.name}: naive ${naiveMs.toFixed(3)} · tiled ${tiledMs.toFixed(3)} ms · identical=${same}`);
      [dims, A, B, bias, C1, C2].forEach((b) => b.destroy());
      await nextFrame();
    }
    // unembed: strided (original) vs coalesced B loads, wte-sized B [50257, 768]
    const V = 50257, D = 768;
    const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dims, 0, new Uint32Array([T, V, D, 0]));
    const A = storage(randomF16(T * D)), B = storage(randomF16(V * D)), C = storage(T * V * 4);
    const grid: [number, number] = [Math.ceil(V / 16), Math.ceil(T / 16)];
    const oldMs = await time(unembedOld, [dims, A, B, C], grid);
    const hOld = await readHash(C, T * V * 4);
    const newMs = await time(unembedNew, [dims, A, B, C], grid);
    const hNew = await readHash(C, T * V * 4);
    rows.push({ T, shape: "unembed 768→50257", stridedMs: oldMs, currentMs: newMs, speedup: oldMs / newMs, bitIdentical: hOld === hNew });
    log(`micro T=${T} unembed: strided ${oldMs.toFixed(3)} · current ${newMs.toFixed(3)} ms · identical=${hOld === hNew}`);
    [dims, A, B, C].forEach((b) => b.destroy());
  }
  return rows;
}

main().catch((e) => {
  log(`ERROR ${(e as Error).stack ?? e}`);
  void fetch("/__bench", { method: "POST", body: JSON.stringify({ label: LABEL, error: String((e as Error).stack ?? e) }) }).catch(() => {});
});
