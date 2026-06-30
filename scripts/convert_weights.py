#!/usr/bin/env python3
"""
Convert HF GPT-2 small weights → single binary shard + manifest.json
(format consumed by src/engine/weights.ts and src/cpu/validate.ts).

    pip install transformers torch
    python scripts/convert_weights.py --dtype f16 --out dist-weights/      # browser
    python scripts/convert_weights.py --dtype f32 --out dist-weights-f32/  # CPU CI

Then upload the f16 output to a HF model repo (free CDN with CORS + Range).

Naming: HF parameter names with the "transformer." prefix stripped
(wte.weight, h.4.attn.c_attn.weight, ln_f.bias, ...).

HF GPT-2 uses Conv1D projection weights are
stored [d_in, d_out] and applied y = x @ W + b. The whole pipeline (manifest →
matmul_tiled.wgsl → cpu/reference.ts) assumes exactly that. No transposes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

# Buffers to EXCLUDE: attn.bias is the causal-mask buffer (not a parameter)
# and lm_head.weight is tied to wte.weight.
EXCLUDE_SUFFIXES = (".attn.bias", ".attn.masked_bias")
EXCLUDE_EXACT = {"lm_head.weight"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--out", default="dist-weights")
    args = ap.parse_args()

    np_dtype = np.dtype("<f2" if args.dtype == "f16" else "<f4")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model.state_dict()

    entries, blobs, offset = [], [], 0
    for name, tensor in sd.items():
        name = name.removeprefix("transformer.")
        if name in EXCLUDE_EXACT or name.endswith(EXCLUDE_SUFFIXES):
            continue
        arr = tensor.detach().to(torch.float32).numpy().astype(np_dtype)
        blob = arr.tobytes()
        entries.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": args.dtype,
            "byteOffset": offset,
            "byteLength": len(blob),
            "shard": "weights.bin",
        })
        blobs.append(blob)
        offset += len(blob)
        if offset % 4:  # 4-byte alignment for WebGPU writeBuffer slices
            pad = 4 - (offset % 4)
            blobs.append(b"\x00" * pad)
            offset += pad

    (out / "weights.bin").write_bytes(b"".join(blobs))
    (out / "manifest.json").write_text(json.dumps(
        {"model": "gpt2", "entries": entries, "totalBytes": offset}, indent=1))
    mb = offset / 1e6
    print(f"Wrote {len(entries)} tensors, {mb:.1f} MB ({args.dtype}) → {out}")


if __name__ == "__main__":
    main()
