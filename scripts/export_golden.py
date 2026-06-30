#!/usr/bin/env python3
"""
Export TransformerLens golden activations for GPT-2 small → fixtures/.

Run locally :
    pip install transformer_lens torch numpy
    python scripts/export_golden.py fixtures/

Output format (consumed by src/cpu/validate.ts):
    fixtures/index.json
    fixtures/<promptId>/<hook>.bin     raw little-endian f32, C-contiguous


 REMINDER:                        
 TransformerLens from_pretrained("gpt2") PROCESSES weights by default      
 (fold_ln, center_writing_weights, center_unembed). That changes the       
 activations — they will never match a faithful raw-GPT-2 implementation.  
 We export with from_pretrained_no_processing so goldens describe the      
 RAW architecture, which is what the engine and CPU reference implement.   

"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

PROMPTS = [
    ("p0", "The Eiffel Tower is in the city of"),
    ("p1", "When Mary and John went to the store, John gave a drink to"),
    ("p2", "The quick brown fox jumps over the lazy"),
    ("p3", "1 2 3 4 5 6 7 8"),
    ("p4", "import numpy as np\nimport torch"),
    ("p5", "Q: What is the capital of France?\nA:"),
    ("p6", "ABC ABC ABC ABC AB"),          # induction-friendly repetition
    ("p7", "Hello world"),                  # short — exercises tiny T
]

# One intervention-level fixture: zero head (9, 9) — an IOI name-mover
# (Wang et al. 2022) — on p1 and save the resulting logits.
ABLATE_LAYER, ABLATE_HEAD, ABLATE_PROMPT = 9, 9, "p1"


def wanted_hooks(n_layers: int) -> list[str]:
    hooks = ["hook_embed", "hook_pos_embed", "ln_final.hook_normalized"]
    for i in range(n_layers):
        p = f"blocks.{i}"
        hooks += [
            f"{p}.hook_resid_pre", f"{p}.hook_resid_mid", f"{p}.hook_resid_post",
            f"{p}.ln1.hook_normalized", f"{p}.ln2.hook_normalized",
            f"{p}.attn.hook_q", f"{p}.attn.hook_k", f"{p}.attn.hook_v",
            f"{p}.attn.hook_attn_scores", f"{p}.attn.hook_pattern",
            f"{p}.attn.hook_z",
            f"{p}.hook_attn_out", f"{p}.hook_mlp_out",
            f"{p}.mlp.hook_pre", f"{p}.mlp.hook_post",
        ]
    return hooks


#write a tensor to disk, record where to find it
def save(out_dir: Path, pid: str, hook: str, tensor: torch.Tensor, index: dict) -> None:
    arr = tensor.detach().to(torch.float32).cpu().numpy()
    rel = f"{pid}/{hook}.bin"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.astype("<f4").tofile(path)
    index["tensors"].append(
        {"prompt": pid, "hook": hook, "shape": list(arr.shape), "dtype": "f32", "file": rel}
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)

    #Download and load GPT-2. Keep dtype float32 end to end.
    model = HookedTransformer.from_pretrained_no_processing("gpt2", dtype=torch.float32)
    model.eval()
    hooks = set(wanted_hooks(model.cfg.n_layers))

    #run model on prompts save activations to disk
    index: dict = {"prompts": [], "tensors": []}
    with torch.no_grad():
        for pid, text in PROMPTS:
            tokens = model.to_tokens(text)  # includes BOS per TL default
            index["prompts"].append(
                {"id": pid, "text": text, "tokens": tokens[0].tolist()}
            )
            logits, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hooks
            )
            for hook in sorted(hooks):
                save(out_dir, pid, hook, cache[hook], index)
            save(out_dir, pid, "logits", logits, index)

        # Intervention-level fixture: ablated logits for (layer, head).
        #head 9 layer 9 is the name-mover head (Wang et al.)
        def zero_head(z: torch.Tensor, hook) -> torch.Tensor:  # z: [b, pos, head, d_head]
            z[:, :, ABLATE_HEAD, :] = 0.0
            return z

        tokens = model.to_tokens(dict(PROMPTS)[ABLATE_PROMPT])
        ablated = model.run_with_hooks(
            tokens,
            fwd_hooks=[(f"blocks.{ABLATE_LAYER}.attn.hook_z", zero_head)],
        )
        save(out_dir, ABLATE_PROMPT,
             f"logits__ablate_L{ABLATE_LAYER}H{ABLATE_HEAD}", ablated, index)

    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Wrote {len(index['tensors'])} tensors for {len(index['prompts'])} prompts → {out_dir}")


if __name__ == "__main__":
    main()
