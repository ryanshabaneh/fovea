# fovea

Delete one attention head from GPT-2 in your browser and watch what it
forgets. A hand-rolled WebGPU forward pass with TransformerLens-style hooks —
no backend.

## Layout

- `src/engine/` — WebGPU engine (weights, kernels, forward pass, hooks)
- `src/kernels/` — WGSL compute kernels
- `src/cpu/` — fp32 reference forward pass + validation harness
- `scripts/` — golden-tensor export and weight conversion
- `ARCHITECTURE.md` — design notes

## Dev

```
npm install
npm run typecheck
npm test
```