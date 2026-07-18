/**
 * Browser engine check (dev tooling, not part of the app).
 * Boots WebGPU, loads weights from the CDN, runs a few prompts through
 * ForwardPass, and prints the greedily-predicted next token. A sensible
 * prediction ("The capital of France is" -> " Paris") means the whole GPU
 * forward pass is wired correctly end to end.
 *
 * Build: npm run build
 * Serve: python3 -m http.server 8000   (from the repo root)
 * Open:  http://localhost:8000/
 */
import { GPT2_SMALL } from "./engine/config.js";
import { WeightStore } from "./engine/weights.js";
import { KernelRegistry } from "./engine/kernels.js";
import { GPT2Tokenizer } from "./engine/tokenizer.js";
import { ForwardPass } from "./engine/forward.js";

const WEIGHTS_URL = "https://huggingface.co/ryanshabaneh/fovea-weights/resolve/main";

const outEl = document.getElementById("out")!;
function log(msg: string, cls = ""): void {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  outEl.appendChild(line);
}

async function main(): Promise<void> {
  // 1. WebGPU device (request shader-f16 if the GPU supports it).
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) { log("WebGPU not available in this browser.", "err"); return; }
  const hasF16 = adapter.features.has("shader-f16");
  // The logits buffer is ~196 MB (n_ctx × 50257 × f32), above the default
  // 128 MB storage-binding limit — request the adapter's higher limits.
  const device = await adapter.requestDevice({
    requiredFeatures: hasF16 ? (["shader-f16"] as GPUFeatureName[]) : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  // Surface GPU validation errors that would otherwise be silent (and leave
  // buffers zero-initialized — exactly the "all zeros" symptom).
  device.addEventListener("uncapturederror", (e) => {
    log("⚠ GPU ERROR: " + (e as GPUUncapturedErrorEvent).error.message, "err");
  });
  log(`device ready — shader-f16: ${hasF16}`);

  // 2. Load weights from the CDN (cached after first visit).
  log("loading weights (~248 MB, cached after first load)…");
  const weights = await WeightStore.load(device, WEIGHTS_URL, (loaded, total) => {
    const pct = ((loaded / total) * 100).toFixed(0);
    outEl.lastElementChild!.textContent = `loading weights… ${pct}%`;
  });
  log("weights on GPU", "ok");

  // 3. Compile kernels, load tokenizer.
  const kernels = new KernelRegistry(device);
  await kernels.loadAll("/src/kernels");
  log("kernels compiled");

  const tokenizer = await GPT2Tokenizer.load("/public/tokenizer");
  log("tokenizer ready");

  // 4. Numerical validation against golden fixtures (TransformerLens) for p0.
  //    Same exact tokens (BOS-prefixed), diff each hook to find the first
  //    kernel that drifts.
  const fwd = new ForwardPass(device, GPT2_SMALL, weights, kernels, tokenizer);

  // p0 = "The Eiffel Tower is in the city of", with the leading BOS <|endoftext|>.
  const p0tokens = new Uint32Array([50256, 464, 412, 733, 417, 8765, 318, 287, 262, 1748, 286]);

  // Hooks to compare (skip ln*.hook_normalized — GPU records post-γβ, golden pre-γβ).
  const probes = [
    "blocks.0.hook_resid_pre",
    "blocks.0.attn.hook_attn_scores",
    "blocks.0.attn.hook_pattern",
    "blocks.0.attn.hook_z",
    "blocks.0.hook_attn_out",
    "blocks.0.hook_resid_mid",
    "blocks.0.mlp.hook_pre",
    "blocks.0.mlp.hook_post",
    "blocks.0.hook_mlp_out",
    "blocks.0.hook_resid_post",
    "blocks.1.hook_resid_pre",
    "blocks.5.hook_resid_post",
    "blocks.11.hook_resid_post",
  ] as const;

  log(`validating vs golden p0 (T=${p0tokens.length})`);
  log("");

  const { logits } = await fwd.run(p0tokens, { record: probes as unknown as string[] as never });

  // fetch a golden .bin as Float32Array
  async function golden(name: string): Promise<Float32Array> {
    const buf = await (await fetch(`/fixtures/p0/${name}.bin`)).arrayBuffer();
    return new Float32Array(buf);
  }

  function diff(gpu: Float32Array, gold: Float32Array) {
    const n = Math.min(gpu.length, gold.length);
    let maxAbs = 0, maxB = 0;
    for (let i = 0; i < n; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(gpu[i] - gold[i]));
      maxB = Math.max(maxB, Math.abs(gold[i]));
    }
    return { maxAbs, maxB, lenMatch: gpu.length === gold.length, gpuN: gpu.length, goldN: gold.length };
  }

  for (const hook of probes) {
    try {
      const gpu = await fwd.cache.readback("default", hook as never);
      const gold = await golden(hook);
      const d = diff(gpu, gold);
      const rel = d.maxAbs / (d.maxB + 1e-9);
      const pass = d.lenMatch && d.maxAbs < 1e-1; // loose f16 tolerance
      const lenNote = d.lenMatch ? "" : `  ⚠ len ${d.gpuN} vs ${d.goldN}`;
      log(`${hook.padEnd(34)} maxAbs=${d.maxAbs.toExponential(2)} (rel ${rel.toExponential(1)})${lenNote}`, pass ? "ok" : "err");
    } catch (e) {
      log(`${hook.padEnd(34)} ${(e as Error).message}`, "err");
    }
  }

  // logits comparison
  try {
    const gold = await golden("logits");
    const d = diff(logits, gold);
    log("");
    log(`logits  maxAbs=${d.maxAbs.toExponential(2)} maxB=${d.maxB.toExponential(2)}  len ${d.gpuN} vs ${d.goldN}`, d.lenMatch && d.maxAbs < 5e-1 ? "ok" : "err");
  } catch (e) {
    log("logits compare: " + (e as Error).message, "err");
  }

  // 5. Definitive check: for ALL 8 golden prompts, does the GPU's top predicted
  //    token match TransformerLens's top token exactly?
  const V = GPT2_SMALL.vocab_size;
  const argmaxLastRow = (a: Float32Array, T: number) => {
    const base = (T - 1) * V;
    let id = 0, best = -Infinity;
    for (let v = 0; v < V; v++) if (a[base + v] > best) { best = a[base + v]; id = v; }
    return id;
  };

  const index = await (await fetch("/fixtures/index.json")).json();
  const prompts: { id: string; text: string; tokens: number[] }[] = index.prompts;

  log("");
  log("=== all-prompts argmax vs golden ===");
  let matches = 0;
  for (const p of prompts) {
    const toks = new Uint32Array(p.tokens);
    const res = await fwd.run(toks);
    const gpuTop = argmaxLastRow(res.logits, toks.length);

    const goldBuf = await (await fetch(`/fixtures/${p.id}/logits.bin`)).arrayBuffer();
    const gold = new Float32Array(goldBuf);
    const goldTop = argmaxLastRow(gold, toks.length);

    const ok = gpuTop === goldTop;
    if (ok) matches++;
    log(`${p.id}  ${JSON.stringify(p.text).slice(0, 40).padEnd(42)} GPU→${JSON.stringify(tokenizer.decode([gpuTop]))}  gold→${JSON.stringify(tokenizer.decode([goldTop]))}  ${ok ? "MATCH" : "MISMATCH"}`, ok ? "ok" : "err");
  }
  log("");
  log(`${matches}/${prompts.length} prompts match TransformerLens exactly`, matches === prompts.length ? "ok" : "err");
}

main().catch((e) => {
  log("ERROR: " + (e as Error).message, "err");
  log((e as Error).stack ?? "", "err");
});