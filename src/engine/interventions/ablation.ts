import type { HookName, HookWrite } from "../types.js";

/** Zeroes hook_z[:, head, :] - the head's output before W_O mixes heads
 *  (not the attention pattern). */
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

  toHookWrite(nHeads = 12): HookWrite {
    if (this.mode === "mean") {
      throw new Error("mean ablation: means fixture not shipped yet");
    }
    const mask = new Float32Array(nHeads).fill(1);
    mask[this.head] = 0;
    return { hook: this.hook, op: { kind: "head_mask", mask } };
  }
}
