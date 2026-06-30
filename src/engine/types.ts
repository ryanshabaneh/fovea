/**
 * Hook-point names use TransformerLens naming. These literal types
 * make a typo a compile error instead of a silent no-op hook.
 */
export type AttnHook =
  | `blocks.${number}.attn.hook_q`
  | `blocks.${number}.attn.hook_k`
  | `blocks.${number}.attn.hook_v`
  | `blocks.${number}.attn.hook_attn_scores`
  | `blocks.${number}.attn.hook_pattern`
  | `blocks.${number}.attn.hook_z`;

export type BlockHook =
  | `blocks.${number}.hook_resid_pre`
  | `blocks.${number}.hook_resid_mid`
  | `blocks.${number}.hook_resid_post`
  | `blocks.${number}.hook_attn_out`
  | `blocks.${number}.hook_mlp_out`
  | `blocks.${number}.ln1.hook_normalized`
  | `blocks.${number}.ln2.hook_normalized`
  | `blocks.${number}.mlp.hook_pre`
  | `blocks.${number}.mlp.hook_post`
  | AttnHook;

export type HookName =
  | "hook_embed"
  | "hook_pos_embed"
  | "ln_final.hook_normalized"
  | BlockHook;

export const WRITABLE_HOOK_SUFFIXES = [
  "attn.hook_z",
  "attn.hook_pattern",
  "hook_resid_pre",
  "hook_resid_mid",
  "hook_resid_post",
] as const;

//guard 
export function isWritableHook(name: HookName): boolean {
  return WRITABLE_HOOK_SUFFIXES.some((s) => name.endsWith(s));
}

export type Dtype = "f16" | "f32" | "u32";

export interface TensorMeta {
  name: string;
  shape: number[];
  dtype: Dtype;
}

/** A write op the HookManager applies to the live buffer at a hook point. */
export type HookWriteOp =
  | { kind: "head_mask"; mask: Float32Array }          // ablation: per-head scale, length n_heads
  | { kind: "copy_from"; src: GPUBuffer; bytes: number }; // patching: overwrite from cached run

export interface HookWrite {
  hook: HookName;
  op: HookWriteOp;
}

export interface RunOptions {
  /** Hooks to read into the ActivationCache (f32 readback happens lazily). */
  record?: HookName[];
  /** Writes applied between producer and consumer kernels. */
  writes?: HookWrite[];
  /** Cache run id; reads are stored under this key. Default "default". */
  runId?: string;
}

export interface RunResult {
  /** fp32 logits, shape [T, vocab_size]. */
  logits: Float32Array;
  seqLen: number;
  runId: string;
}
