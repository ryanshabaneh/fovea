import { GPT2_SMALL, type ModelConfig } from "./config.js";
import { WeightStore } from "./weights.js";
import { KernelRegistry } from "./kernels.js";
import { GPT2Tokenizer } from "./tokenizer.js";
import { ForwardPass } from "./forward.js";

/** Public HuggingFace CDN holding the fp16 weight shard + manifest. */
export const DEFAULT_WEIGHTS_URL =
  "https://huggingface.co/ryanshabaneh/fovea-weights/resolve/main";

export type BootStage = "adapter" | "weights" | "kernels" | "tokenizer" | "ready";

export interface BootProgress {
  stage: BootStage;
  /** Bytes loaded so far (weights stage only). */
  loaded?: number;
  /** Total bytes (weights stage only). */
  total?: number;
  /** Human-readable detail for a boot console (real, not synthesized). */
  note?: string;
}

export interface BootOptions {
  weightsUrl?: string;
  /** Where the engine fetches `<name>.wgsl`. Default "/kernels". */
  kernelsPath?: string;
  /** Where the engine fetches encoder.json / vocab.bpe. Default "/tokenizer". */
  tokenizerPath?: string;
  onProgress?: (p: BootProgress) => void;
}

export interface BootResult {
  device: GPUDevice;
  cfg: ModelConfig;
  fwd: ForwardPass;
  tokenizer: GPT2Tokenizer;
  hasF16: boolean;
}

/**
 * One-time engine startup: WebGPU device (with the raised buffer limits the
 * forward pass needs), weight download, kernel compile, tokenizer. Returns a
 * live ForwardPass ready for run()/generate(). Extracted verbatim from the
 * validate_browser harness so the UI and the harness boot identically.
 */
export async function boot(opts: BootOptions = {}): Promise<BootResult> {
  const weightsUrl = opts.weightsUrl ?? DEFAULT_WEIGHTS_URL;
  const kernelsPath = opts.kernelsPath ?? "/kernels";
  const tokenizerPath = opts.tokenizerPath ?? "/tokenizer";
  const report = (p: BootProgress): void => opts.onProgress?.(p);

  // 1. WebGPU device (request shader-f16 if the GPU supports it).
  report({ stage: "adapter", note: "requesting WebGPU adapter" });
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU not available in this browser.");
  const hasF16 = adapter.features.has("shader-f16");
  report({ stage: "adapter", note: `adapter acquired · shader-f16 ${hasF16 ? "supported" : "unavailable"}` });

  // The logits buffer is ~196 MB (n_ctx × 50257 × f32), above the default
  // 128 MB storage-binding limit. Without these raised limits the entire
  // forward pass silently fails and produces all zeros (no exception thrown).
  const device = await adapter.requestDevice({
    requiredFeatures: hasF16 ? (["shader-f16"] as GPUFeatureName[]) : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const maxBufMiB = Math.round(Number(adapter.limits.maxStorageBufferBindingSize) / (1024 * 1024));
  report({ stage: "adapter", note: `device ready · max storage buffer ${maxBufMiB} MiB` });

  // 2. Weights - streamed from the CDN, cached by the browser after first load.
  report({ stage: "weights", loaded: 0, total: 1, note: "fetching manifest + weights.bin" });
  const weights = await WeightStore.load(device, weightsUrl, (loaded, total) => {
    report({ stage: "weights", loaded, total });
  });
  report({ stage: "kernels", note: "weights uploaded to GPU buffers" });

  // 3. Kernels - fetch + compile all WGSL pipelines.
  report({ stage: "kernels", note: "compiling WGSL compute kernels" });
  const kernels = new KernelRegistry(device);
  await kernels.loadAll(kernelsPath);

  // 4. Tokenizer - GPT-2 byte-level BPE from vendored assets.
  report({ stage: "tokenizer", note: "loading BPE tokenizer" });
  const tokenizer = await GPT2Tokenizer.load(tokenizerPath);
  report({ stage: "tokenizer", note: `tokenizer ready · ${GPT2_SMALL.vocab_size} tokens` });

  const fwd = new ForwardPass(device, GPT2_SMALL, weights, kernels, tokenizer);
  report({ stage: "ready", note: "engine ready" });
  return { device, cfg: GPT2_SMALL, fwd, tokenizer, hasF16 };
}