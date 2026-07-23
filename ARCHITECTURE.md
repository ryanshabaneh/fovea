# Architecture

How fovea runs GPT-2 small (124M) in the browser: fp16 weights and activations, fp32 accumulation, no runtime dependencies in the engine, and TransformerLens hook names throughout. Three interventions build on top of it, head ablation, activation patching, and a logit lens.

The engine purpose is to run the forward pass while letting you read or edit any activation along the way. Everything else is built on that.

---

## 1. Directory structure

```
fovea/
├── ARCHITECTURE.md            ← this file
├── README.md
├── LATER.md                   ← ideas for later
├── package.json
├── tsconfig.json
├── public/
│   └── tokenizer/             ← vendored encoder.json + vocab.bpe (static assets)
├── scripts/
│   ├── convert_weights.py     ← HF fp32 safetensors → fp16 shards + manifest.json
│   └── export_golden.py       ← TransformerLens golden activations → fixtures/ (8 prompts for testing)
├── fixtures/                  ← golden tensors (gitignored; ~50 MB) (8 prompts activations)
├── src/
│   ├── engine/
│   │   ├── config.ts          ← ModelConfig interface + GPT2_SMALL const
│   │   ├── types.ts           ← HookName literal types, tensor metadata
│   │   ├── tokenizer.ts       ← GPT-2 byte-level BPE (vendored assets)
│   │   ├── weights.ts         ← WeightStore: CDN → Cache API → GPU buffers
│   │   ├── kernels.ts         ← KernelRegistry: WGSL → compute pipelines
│   │   ├── forward.ts         ← ForwardPass — runs the whole forward pass
│   │   ├── hooks.ts           ← HookManager: read/write at named points
│   │   ├── cache.ts           ← ActivationCache: GPU buffers + readbacks (get activations from GPU back to JS)
│   │   └── interventions/
│   │       ├── ablation.ts    ← HeadAblation
│   │       ├── patch.ts       ← ActivationPatch
│   │       └── logitlens.ts   ← LogitLens
│   ├── kernels/               ← 8 .wgsl files (inventory in §3)
│   ├── cpu/
│   │   ├── reference.ts       ← fp32 ground-truth forward pass (Node-safe)
│   │   ├── validate.ts        ← CPU↔GPU↔golden diffing
│   │   └── reference.test.ts  ← invariant sanity tests (run in CI)
│   └── ui/                    ← the interface (vanilla TS, not covered here)
```

---

## 2. Engine module design

```
                 ┌────────────────────────────────────────────────┐
                 │                  ForwardPass                   │
                 │  run(tokens, {hooks?, record?}) → RunResult    │
                 └──────┬───────────────┬───────────────┬─────────┘
                        │ dispatches    │ fires         │ stores reads
                        ▼               ▼               ▼
              ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
              │KernelRegistry│  │ HookManager │  │ ActivationCache  │
              │ .wgsl→pipes  │  │ read/write  │  │ GPU bufs + f32   │
              └──────┬───────┘  │ at hook pts │  │ readbacks        │
                     │          └──────┬──────┘  └────────▲─────────┘
        reads weights│                 │ write ops copy/  │
                     ▼                 │ mask GPU buffers │ patch source
              ┌──────────────┐         └──────────────────┘
              │ WeightStore  │
              │ manifest →   │   Interventions (HeadAblation, ActivationPatch,
              │ GPU buffers  │   LogitLens) never touch buffers directly — they
              └──────────────┘   compile themselves into HookManager read/write
                                 registrations + readback requests.
```

Data flow for one forward pass (no interventions):

```
tokens(u32) ─▶ embed.wgsl ─▶ resid_pre ─┐
        ┌───────────────────────────────┘  ×12 blocks
        ▼
  layernorm ─▶ matmul(W_qkv) ─▶ split q/k/v ─▶ matmul(qkᵀ·scale)
        ─▶ softmax_causal ─▶ matmul(pattern·v) = z ─▶ [head_zero if ablating]
        ─▶ matmul(W_O) ─▶ residual_add → resid_mid
        ─▶ layernorm ─▶ matmul(W_in) ─▶ gelu ─▶ matmul(W_out)
        ─▶ residual_add → resid_post
        └──────────────────────────────────┐
                                           ▼
                       layernorm(ln_final) ─▶ unembed ─▶ logits (f32)
```

Every named arrow endpoint is a hook point. `HookManager.fire()` is invoked between kernel dispatches; a registered **write** mutates the live buffer before the next kernel consumes it, a registered **read** copies it into `ActivationCache`.

---

## 3. WGSL kernel inventory

Binding convention (all kernels): `@group(0) @binding(0)` = uniform `Dims` struct; inputs ascending from binding 1; output is the last binding, `var<storage, read_write>`. Activations f16 in storage, **all interior arithmetic f32**. The device requests the `"shader-f16"` feature; if it's missing, a build-time string swap turns f16 storage into f32 — 2× the memory, but a single code path.

| # | File | Inputs (shape) | Output (shape) | Bindings | Precision notes |
|---|---|---|---|---|---|
| 1 | `embed.wgsl` | ids u32 `[T]`, wte f16 `[50257,768]`, wpe f16 `[1024,768]` | resid f16 `[T,768]` | 0:Dims{T,D} 1:ids 2:wte 3:wpe 4:out | None numeric. Fuses token+positional add (one pass). Clamp id < vocab. |
| 2 | `layernorm.wgsl` | x f16 `[rows,768]`, γ f16 `[768]`, β f16 `[768]` | y f16 `[rows,768]` | 0:Dims{rows,D,eps} 1:x 2:γ 3:β 4:y | **Variance in fp16 cancels badly.** Two-pass (mean, then Σ(x−μ)²) in f32 workgroup reduction. eps=1e-5 **inside** the sqrt: `(x−μ)/sqrt(σ²+ε)`. |
| 3 | `matmul_tiled.wgsl` | A f16 `[M,K]`, B f16 `[K,N]`, bias f16 `[N]` (flag) | C f16 `[M,N]` | 0:Dims{M,N,K,flags} 1:A 2:B 3:bias 4:C | **Long-K reductions drift in fp16** (K=768/3072). Accumulator is `f32`; convert at load, round once at store. |
| 4 | `gelu.wgsl` | x f16 `[n]` | y f16 `[n]` | 0:Dims{n} 1:x 2:y | Must be **gelu_new** (tanh approx, c=0.044715, √(2/π)=0.7978845608) — the erf variant diverges from GPT-2 goldens at ~1e-3. Math in f32. |
| 5 | `softmax_causal.wgsl` | scores f16 `[12,T,T]`, scale | pattern f16 `[12,T,T]` | 0:Dims{H,T,scale} 1:scores 2:pattern | **Subtract row-max before exp** (f32). Max taken over *valid* (j≤q) positions only; masked positions written as exact 0, never exp(−1e9). |
| 6 | `residual_add.wgsl` | a f16 `[n]`, b f16 `[n]` | out f16 `[n]` | 0:{n} 1:a 2:b 3:out | Add in f32, single round on store (matches CPU op order). |
| 7 | `head_zero.wgsl` | z f16 `[T,12,64]`, mask f32 `[12]` | z' f16 `[T,12,64]` | 0:{T,H,Dh} 1:z 2:mask 3:out | None. Per-head scalar multiply (0 = ablate, 1 = keep; fractional = soft ablation for free). Head index `h = (i / Dh) % H`. |
| 8 | `unembed.wgsl` | resid f16 `[T,768]`, wte f16 `[50257,768]` (read transposed) | logits **f32** `[T,50257]` | 0:{T,D,V} 1:resid 2:wte 3:logits | Weight tying: B is wte accessed as `wte[col*D + k]` (no materialized transpose). **Logits stored f32** — downstream softmax/top-k must not eat an fp16 round. |

Where each matmul instance appears per block: `W_qkv` (768→2304, bias), `qkᵀ` (per-head, scale folded into softmax uniform), `pattern·v`, `W_O` (768→768, bias), `W_in` (768→3072, bias), `W_out` (3072→768, bias). The qkᵀ and pattern·v products are batched-per-head matmuls; each is dispatched as H independent tiled matmuls. A single batched kernel would be faster — noted for later.

---

## 4. Hook point spec (TransformerLens naming, verbatim)

Shapes for GPT-2 small at sequence length T, batch=1 (batch dim elided in buffers, present in golden fixtures as 1).

| Hook name | Tensor | Shape | R/W |
|---|---|---|---|
| `hook_embed` | token embeddings | `[T, 768]` | R |
| `hook_pos_embed` | positional embeddings | `[T, 768]` | R |
| `blocks.{i}.hook_resid_pre` | residual stream into block | `[T, 768]` | **R+W** |
| `blocks.{i}.ln1.hook_normalized` | post-LN1 | `[T, 768]` | R |
| `blocks.{i}.attn.hook_q` | queries | `[T, 12, 64]` | R |
| `blocks.{i}.attn.hook_k` | keys | `[T, 12, 64]` | R |
| `blocks.{i}.attn.hook_v` | values | `[T, 12, 64]` | R |
| `blocks.{i}.attn.hook_attn_scores` | scaled qkᵀ, pre-softmax | `[12, T, T]` | R |
| `blocks.{i}.attn.hook_pattern` | post-softmax attention | `[12, T, T]` | **R+W** |
| `blocks.{i}.attn.hook_z` | per-head outputs, **before W_O** | `[T, 12, 64]` | **R+W** ← ablation lives here |
| `blocks.{i}.hook_attn_out` | after W_O | `[T, 768]` | R |
| `blocks.{i}.hook_resid_mid` | resid_pre + attn_out | `[T, 768]` | **R+W** |
| `blocks.{i}.ln2.hook_normalized` | post-LN2 | `[T, 768]` | R |
| `blocks.{i}.mlp.hook_pre` | pre-GELU | `[T, 3072]` | R |
| `blocks.{i}.mlp.hook_post` | post-GELU | `[T, 3072]` | R |
| `blocks.{i}.hook_mlp_out` | MLP output | `[T, 768]` | R |
| `blocks.{i}.hook_resid_post` | resid_mid + mlp_out | `[T, 768]` | **R+W** |
| `ln_final.hook_normalized` | post final LN | `[T, 768]` | R |

i ∈ [0, 11]. The **write set** is exactly `{hook_z, hook_pattern, hook_resid_pre, hook_resid_mid, hook_resid_post}` — enough for head ablation and every standard patching experiment, and small enough to test exhaustively.

**Key invariant:** ablating head (l,h) means zeroing `blocks.{l}.attn.hook_z[:, h, :]` — the head's 64-dim output vector *before* the W_O projection mixes heads. It is *not* zeroing the attention pattern; a zeroed pattern row still leaks the value bias and renormalizes differently.

**Golden-comparison caveat:** TransformerLens conventions for `hook_attn_scores` at masked positions (−∞ vs large-negative) vary by version. The validation harness compares scores and pattern **on the causal lower triangle only** (j ≤ q). Everything else compares full tensors.

---

## 5. Activation patching, step by step

Goal: run prompt B, but at hook H substitute the activation recorded from prompt A. Constraint: `tokens_A` and `tokens_B` must be the same length (enforced in `ActivationPatch`; aligning different lengths is left for later).

1. **Source run.** `forward.run(tokens_A, { record: [H] })`. At H's fire point, HookManager issues `copyBufferToBuffer(live_H → cache["A"][H])` into a fresh buffer created with `COPY_DST | COPY_SRC`. Nothing else is retained.
2. **Patched run.** `forward.run(tokens_B, { writes: [{hook: H, op: "copy_from", src: cache["A"][H]}] })`.
3. At H's fire point in run B, *after* the producer kernel has written the live buffer and *before* any consumer kernel is encoded, HookManager encodes `copyBufferToBuffer(cache["A"][H] → live_H)` (full overwrite; byte sizes are equal because T is equal).
4. Execution order within the command encoder is therefore: `producer → [reads for recording] → write op → consumer`. Reads registered at H during run B observe the **patched** value (matches TransformerLens semantics, where a read hook downstream of a write hook sees the edit).
5. Both clean-B logits (from an earlier plain run) and patched-B logits are read back f32; the UI diffs token-level argmax/probabilities.

No CPU round-trip occurs mid-forward in any of this; patching is GPU buffer-to-buffer copies inside the same submission.

Ablation is the degenerate case: instead of a copy at `hook_z`, dispatch `head_zero.wgsl` with `mask[h]=0` between producer and consumer.

---

## 6. Weight loading pipeline

```
offline (scripts/convert_weights.py, run once):
  HF gpt2 checkpoint (fp32) ─cast→ fp16 ─▶ weights.bin (single shard, ~248 MB)
                                        └▶ manifest.json: [{name, shape, dtype,
                                            byteOffset, byteLength, shard}]
  → uploaded to a HF model repo (free CDN, supports Range + CORS)

browser (WeightStore.load):
  1. fetch manifest.json
  2. caches.open("poke-weights-v1") → match(shard URL)
     ├─ hit  → Response from Cache API (instant repeat visits)
     └─ miss → fetch with ReadableStream reader → onProgress(bytes/total)
               → cache.put(url, response.clone())
  3. ArrayBuffer → for each manifest entry:
       device.createBuffer({size, usage: STORAGE | COPY_DST})
       device.queue.writeBuffer(buf, 0, bytes, byteOffset, byteLength)
  4. WeightStore.getBuffer("transformer.h.4.attn.c_attn.weight") → GPUBuffer
```

Notes:
- Names in the manifest keep **HF naming and HF Conv1D orientation**: GPT-2's `c_attn`/`c_fc`/etc. store W as `[d_in, d_out]` applied as `y = x @ W + b`. The matmul kernel consumes exactly that layout — **no transposes anywhere**. (Classic porting bug: treating these as nn.Linear `[d_out, d_in]`.)
- One shard, not many: HTTP/2 makes a single 248 MB stream with one progress bar simpler and no slower than parallel shards. Revisit only if a CDN object cap forces it.
- Cache API over OPFS: equivalent persistence for our access pattern, far simpler types, and `Response` streaming for free. OPFS noted as an alternative in `weights.ts` if Cache API eviction proves aggressive on mobile Safari.

---

## 7. CPU reference implementation

`src/cpu/reference.ts` is the **oracle**. Plain TypeScript, fp32 `Float32Array` throughout, zero browser APIs, runs in Node. It is structured per-op (`layerNorm`, `geluNew`, `linear`, `attention`, `mlp`, `block`, `forward`) so each WGSL kernel has a 1:1 CPU counterpart to diff against.

**Numerical identity requirements (CPU ↔ WGSL):**
- Same op order: pre-LN → attn → add → pre-LN → MLP → add. Residual adds happen in the same sequence position.
- LayerNorm: biased variance, ε = 1e-5 inside sqrt, two-pass mean/var.
- GELU: tanh approximation (gelu_new) with identical constants.
- Softmax: scale → causal mask → subtract valid-max → exp → normalize; masked entries exact 0.
- Logits: `ln_final(resid) @ wteᵀ`, **no bias** (weight tying).

**Tolerances (fp16 GPU vs fp32 CPU), per kernel:**

| Op | atol | rtol | Notes |
|---|---|---|---|
| embed, residual_add, gelu, head_zero | 1e-3 | 1e-3 | single fp16 round |
| layernorm | 5e-3 | 1e-2 | reduction + rsqrt |
| matmul (K=768) | 1e-2 | 1e-2 | f32 acc, fp16 I/O rounds |
| matmul (K=3072), unembed | 2e-2 | 1e-2 | longer reduction |
| softmax_causal (pattern) | 1e-3 | — | lower triangle only |
| end-to-end, any hook | **1e-2** | 1e-2 | the end-to-end target |
| final logits | 5e-2 | — | plus: top-5 token sets must match on all golden prompts |

Pass rule: `max(|a−b|) ≤ atol + rtol·max(|b|)` per tensor; report worst element and its index on failure.

**Three-way validation, two environments:**
1. **CI (Node, headless, every push):** CPU reference vs TransformerLens golden fixtures, key-for-key over §4's hook list, tight tolerance (atol 1e-4 — fp32 vs fp32, only op-order noise). This catches math bugs with zero GPU involvement.
2. **Browser test page (manual, plus optional Puppeteer `--enable-unsafe-webgpu` job):** WGSL path vs CPU path on the same prompts, per-kernel then end-to-end, tolerances above. `validate.ts#validateAgainstGPU(hookPoint, atol)` loads both sides and diffs.
3. **Intervention-level:** ablate head (l,h) in TransformerLens and in both local paths; final logits must agree within the logits tolerance. This validates the intervention path end-to-end.