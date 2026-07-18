import type { HookName, HookWrite } from "../types.js";

/**
 * HeadAblation 
 * zeroes z = blocks.{layer}.attn.hook_z[:, head, :], the head's 64-dim output
 * vector for every position, AFTER attention is computed and BEFORE the
 * shared W_O projection mixes heads. It does NOT zero the attention pattern;
 * a zeroed pattern row would still leak v-bias and renormalize differently.
 */
export class HeadAblation {
  constructor(
    readonly layer: number,
    readonly head: number,
    readonly mode: "zero" | "mean" = "zero",
  ) {
    if (layer < 0 || layer > 11) throw new Error(`layer ${layer} out of range for GPT-2 small`);
    if (head < 0 || head > 11) throw new Error(`head ${head} out of range for GPT-2 small`);
  }

  get hook(): HookName {
    return `blocks.${this.layer}.attn.hook_z` as HookName;
  }

  /** Compile to a HookManager write op (head_zero.wgsl dispatch with this mask). */
  toHookWrite(nHeads = 12): HookWrite {
    if (this.mode === "mean") {
      throw new Error("mean ablation: means fixture not shipped yet");
    }
    const mask = new Float32Array(nHeads).fill(1);
    mask[this.head] = 0;
    return { hook: this.hook, op: { kind: "head_mask", mask } };
  }
}
