import type { HookName, HookWrite } from "../types.js";
import { isWritableHook } from "../types.js";
import type { ActivationCache } from "../cache.js";
import { HookManager } from "../hooks.js";
import type { ModelConfig } from "../config.js";

/** Substitute a hook's activation from a recorded run into another run. */
export class ActivationPatch {
  constructor(
    readonly sourceRunId: string,
    readonly hook: HookName,
  ) {
    if (!isWritableHook(hook))
      throw new Error(`${hook} is not patchable in v1 (write set: ARCHITECTURE.md §4)`);
  }

  /** Source and target seq lengths must match, it's a byte-for-byte overwrite. */
  toHookWrite(cache: ActivationCache, cfg: ModelConfig, targetSeqLen: number): HookWrite {
    const srcShape = cache.getShape(this.sourceRunId, this.hook);
    const expected = HookManager.hookShape(this.hook, cfg, targetSeqLen);
    if (srcShape.join("x") !== expected.join("x")) {
      throw new Error(
        `Patch shape mismatch at ${this.hook}: source [${srcShape}] vs target [${expected}]. ` +
        `v1 requires equal token counts in both prompts.`,
      );
    }
    const bytes = srcShape.reduce((a, b) => a * b, 1) * 2; // f16
    return {
      hook: this.hook,
      op: { kind: "copy_from", src: cache.getBuffer(this.sourceRunId, this.hook), bytes },
    };
  }
}
