<div align="center">

# fovea

**Delete an attention head from GPT-2, right in your browser, and watch what it forgets.**

[**▶ Live demo**](https://fovea-gpt2.vercel.app/) · desktop Chrome or Edge

</div>

---

## What you can do

- **Run a prompt** and see GPT-2's next token guesses, with probabilities.
- **Switch off any of its 144 attention heads** and measure how much that changes the prediction.
- **Click a head** to see which words it's actually paying attention to.

It's a small tool for *mechanistic interpretability*, figuring out what individual parts of a network do, instead of treating it as a black box.

## How it works

Fovea runs GPT-2 small (124M parameters) entirely on the GPU. The complete forward pass — embeddings, multi-head self-attention, MLPs, and layer normalization — is implemented from scratch as **WGSL compute shaders**, with activations stored in fp16 and accumulated in fp32 to stay numerically faithful across long reductions.

- **Nothing runs on a server.** The model's fp16 weights (~248 MB) stream directly from Hugging Face into GPU memory and are cached locally for instant repeat visits. All inference happens client-side; no data leaves your browser.
- **Verification.** Each GPU kernel is validated against a readable fp32 CPU implementation of GPT-2, and end-to-end against activations exported from [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), agreeing to within a documented per-kernel tolerance. The CPU model is the oracle; the WGSL path must match it op for op.
- **Activations are exposed through hooks.** Following TransformerLens's hook-point naming, the engine can read or edit any intermediate activation between kernel dispatches. This is the mechanism behind every intervention: ablating an attention head, for example, zeroes its output vector at `hook_z` before the projection that mixes heads together.

## Run it locally

```bash
npm install
npm run dev 
```

```bash
npm test          # checks the GPU math against the CPU reference
```

**Requirements:** a browser with WebGPU — desktop Chrome or Edge (114+). Safari, Firefox, and phones aren't there yet.

## Project layout

```
src/engine/    WebGPU engine — weights, kernels, forward pass, hooks
src/kernels/   the WGSL GPU kernels
src/cpu/       the CPU reference model everything is checked against
src/ui/        the interface
```

## License

MIT