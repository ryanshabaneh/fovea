/**
 * Invariant tests for the CPU reference, on a seeded micro config with random
 * weights 
 *
 * These prove mathematical INVARIANTS (causality, normalization, ablation
 * semantics), not GPT-2 correctness. Correctness against the real model is
 * validateCpuAgainstGolden() with fixtures  (scripts/export_golden.py).
 *
 * Run: npm test   (tsc build, then node dist/cpu/reference.test.js)
 */

//grab fns from refernce.ts that the tests call
import {
  forward, geluNew, layerNorm, randomWeights, rng,
  type CpuHookEdit,
} from "./reference.js";
import { compareTensors } from "./validate.js";
import { HookManager } from "../engine/hooks.js"; //for the .hookShape()
import type { ModelConfig } from "../engine/config.js";
import type { HookName } from "../engine/types.js";

const MICRO: ModelConfig = {
  n_layers: 2, n_heads: 2, d_model: 8, d_head: 4,
  d_ff: 16, vocab_size: 11, n_ctx: 8, ln_eps: 1e-5,
};

let failures = 0;
function test(name: string, fn: () => void): void {
  try {
    fn();
    console.log(`  PASS  ${name}`);
  } catch (err) {
    failures++;
    console.error(`  FAIL  ${name}\n        ${(err as Error).message}`);
  }
}
function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

//checks that a and b are within tol of each other.

function approx(a: number, b: number, tol: number, what: string): void {
  assert(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);
}

const W = randomWeights(MICRO, 42); //makes the fake weight dict, 42 seed ensures same weights every run

//creates a fake res stream and apply layernorm -> recomputes mean and variance manually from the output,
// then checks they're close to 0 and 1.

test("layerNorm: unit gamma/zero beta gives mean≈0, var≈1 per row", () => {
  const D = 32, rows = 4;
  const r = rng(7);
  const x = new Float32Array(rows * D).map(() => r() * 4 - 2);
  const y = layerNorm(x, rows, D, new Float32Array(D).fill(1), new Float32Array(D), 1e-5);
  for (let row = 0; row < rows; row++) {
    let mean = 0;
    for (let j = 0; j < D; j++) mean += y[row * D + j];
    mean /= D;
    let v = 0;
    for (let j = 0; j < D; j++) v += (y[row * D + j] - mean) ** 2;
    v /= D;
    approx(mean, 0, 1e-6, `row ${row} mean`);
    approx(v, 1, 1e-3, `row ${row} var`); // eps slightly shrinks var below 1
  }
});

test("geluNew: matches tanh-approximation reference values", () => {
  const y = geluNew(new Float32Array([0, 1, -1, 5, -5]));
  approx(y[0], 0, 1e-9, "gelu(0)");
  approx(y[1], 0.841192, 1e-4, "gelu(1)");   // tanh approx — NOT the erf value 0.841345 (they diverge at x=1)
  approx(y[2], -0.158815, 1e-4, "gelu(-1)");
  approx(y[1] - y[2], 1, 1e-4, "gelu(1) - gelu(-1) = 1 (x·Φ(x) + x·Φ(-x) = x)");
  approx(y[3], 5, 1e-3, "gelu(5)≈5");
  approx(y[4], 0, 1e-3, "gelu(-5)≈0");
});

test("attention pattern: causal zeros above diagonal, valid rows sum to 1", () => {
  const tokens = [1, 2, 3, 4, 5];
  const T = tokens.length, H = MICRO.n_heads;
  const { cache } = forward(tokens, MICRO, W, { record: ["blocks.0.attn.hook_pattern"] });
  const pat = cache.get("blocks.0.attn.hook_pattern");
  assert(!!pat, "pattern not recorded");
  const p = pat!.data;
  for (let h = 0; h < H; h++)
    for (let q = 0; q < T; q++) {
      let sum = 0;
      for (let k = 0; k < T; k++) {
        const v = p[(h * T + q) * T + k];
        if (k > q) assert(v === 0, `pattern[${h},${q},${k}] above diagonal must be exact 0, got ${v}`);
        else { assert(v >= 0, "negative probability"); sum += v; }
      }
      approx(sum, 1, 1e-5, `row sum h=${h} q=${q}`);
    }
});

test("CAUSALITY: perturbing a later token leaves earlier logits bit-identical", () => {
  const a = forward([1, 2, 3, 4], MICRO, W).logits;
  const b = forward([1, 2, 3, 9], MICRO, W).logits;
  const V = MICRO.vocab_size;
  for (let t = 0; t < 3; t++)
    for (let v = 0; v < V; v++)
      assert(a[t * V + v] === b[t * V + v],
        `logits[${t},${v}] changed when only token 3 differed`);
  // and position 3 itself must change:
  let diff = 0;
  for (let v = 0; v < V; v++) diff = Math.max(diff, Math.abs(a[3 * V + v] - b[3 * V + v]));
  assert(diff > 1e-6, "last-position logits should differ");
});

test("ablation semantics: all-ones mask is a no-op; zero mask changes logits", () => {
  const tokens = [3, 1, 4, 1, 5];
  const hook = "blocks.0.attn.hook_z";
  const clean = forward(tokens, MICRO, W).logits;

  const maskWith = (m: number): CpuHookEdit => (data, shape) => {
    const [T, H, Dh] = shape;
    for (let i = 0; i < T * H * Dh; i++) {
      const h = ((i / Dh) | 0) % H;
      if (h === 0) data[i] *= m; // ablate head 0
    }
  };

  const noop = forward(tokens, MICRO, W, { edits: { [hook]: maskWith(1) } }).logits;
  const r1 = compareTensors(noop, clean, { atol: 0 });
  assert(r1.pass, `ones-mask changed logits (maxAbs ${r1.maxAbs})`);

  const ablated = forward(tokens, MICRO, W, { edits: { [hook]: maskWith(0) } }).logits;
  const r2 = compareTensors(ablated, clean, { atol: 0 });
  assert(!r2.pass && r2.maxAbs > 1e-6, "zero-mask should change logits");
});

test("patching semantics: full resid_pre overwrite at block 1 transplants run A's logits", () => {
  // Everything downstream of blocks.1.hook_resid_pre depends ONLY on that
  // tensor — so patching all of it from run A must reproduce A's logits
  // exactly. This is the strongest cheap test of HookWrite ordering.
  const tokA = [1, 2, 3, 4], tokB = [9, 8, 7, 6]; // equal length (v1 constraint)
  const hook = "blocks.1.hook_resid_pre";
  const runA = forward(tokA, MICRO, W, { record: [hook] });
  const src = runA.cache.get(hook)!.data;
  const patched = forward(tokB, MICRO, W, {
    edits: { [hook]: (data) => data.set(src) },
  }).logits;
  const r = compareTensors(patched, runA.logits, { atol: 0 });
  assert(r.pass, `patched-B logits != A logits (maxAbs ${r.maxAbs})`);
});

test("recorded shapes agree with HookManager.hookShape (engine ↔ oracle contract)", () => {
  const tokens = [1, 2, 3, 4, 5, 6];
  const { cache } = forward(tokens, MICRO, W, { record: "all" });
  assert(cache.size > 30, `expected >30 recorded hooks, got ${cache.size}`);
  for (const [name, t] of cache) {
    const expected = HookManager.hookShape(name as HookName, MICRO, tokens.length);
    assert(
      t.shape.join("x") === expected.join("x"),
      `${name}: recorded [${t.shape}] vs hookShape [${expected}]`,
    );
  }
});

console.log(failures === 0 ? "\nAll invariant tests passed." : `\n${failures} test(s) FAILED.`);
process.exitCode = failures === 0 ? 0 : 1;
