import type { HookName, HookWrite } from "../types.js";
import { isWritableHook } from "../types.js";
import type { ActivationCache } from "../cache.js";
import { HookManager } from "../hooks.js";
import type { ModelConfig } from "../config.js";

/**
 * ActivationPatch — run prompt B with hook H's activation substituted from a
 * recorded run of prompt A. This class only validates and compiles the write op.
 */
export class ActivationPatch {
  constructor(
    readonly sourceRunId: string,
    readonly hook: HookName,
  ) {
    if (!isWritableHook(hook))
      throw new Error(`${hook} is not patchable in v1 (write set: ARCHITECTURE.md §4)`);
  }

  /** 
   * source and target sequence lengths must be equal, 
   * otherwise the byte-for-byte buffer overwrite is positionally meaningless. 
   */
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
