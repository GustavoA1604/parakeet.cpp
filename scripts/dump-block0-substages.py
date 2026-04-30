#!/usr/bin/env python3
"""Dump per-sub-stage outputs inside block 0 from the Python shadow.

Produces artifacts/ctc-ref/block_0_{post_ff1,post_attn,post_conv,post_ff2}.npy
so the C++ test-encoder can isolate which sub-component diverges.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
shadow = __import__("ref-encoder-from-gguf".replace("-", "_")) if False else None
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ref_shadow",
    str(Path(__file__).parent / "ref-encoder-from-gguf.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, default=Path("models/parakeet-ctc-0.6b.gguf"),
                   help="CTC GGUF whose weights drive the Python shadow encoder.")
    p.add_argument("--out",  type=Path, default=Path("artifacts/ctc-ref"),
                   help="Directory the .npy substage dumps are written to "
                        "(must already contain the mel.npy from dump-ctc-reference.py).")
    args = p.parse_args()
    out = args.out
    W, meta = mod.load_gguf(str(args.gguf))
    mel = torch.from_numpy(np.load(out / "mel.npy"))
    mel_valid = int((mel != 0).any(dim=0).sum().item())

    d_model = meta["parakeet.encoder.d_model"]
    n_heads = meta["parakeet.encoder.n_heads"]

    with torch.inference_mode():
        x_sub, _ = mod.subsampling(mel, W, valid_len=mel_valid)
        x = x_sub * math.sqrt(d_model)
        T = x.size(1)
        pe_full = mod.sinusoidal_rel_pe(max(T, 5000), d_model, dtype=x.dtype)
        center = pe_full.size(1) // 2 + 1
        pos_emb = pe_full[:, center - T : center + T - 1]

        p = "encoder.blk.0"
        residual = x
        y = mod.layer_norm(x, W[f"{p}.norm_ff1.weight"], W[f"{p}.norm_ff1.bias"])
        y = mod.conformer_ff(y, W, f"{p}.ff1")
        post_ff1 = residual + 0.5 * y

        residual = post_ff1
        y = mod.layer_norm(post_ff1, W[f"{p}.norm_attn.weight"], W[f"{p}.norm_attn.bias"])
        y = mod.rel_pos_mha(y, pos_emb, W, f"{p}.attn", n_heads)
        post_attn = residual + y

        residual = post_attn
        y = mod.layer_norm(post_attn, W[f"{p}.norm_conv.weight"], W[f"{p}.norm_conv.bias"])
        y = mod.conformer_conv(y, W, f"{p}.conv")
        post_conv = residual + y

        residual = post_conv
        y = mod.layer_norm(post_conv, W[f"{p}.norm_ff2.weight"], W[f"{p}.norm_ff2.bias"])
        y = mod.conformer_ff(y, W, f"{p}.ff2")
        post_ff2 = residual + 0.5 * y

        block_0_out = mod.layer_norm(post_ff2, W[f"{p}.norm_out.weight"], W[f"{p}.norm_out.bias"])

    for name, t in [
        ("block_0_post_ff1",  post_ff1),
        ("block_0_post_attn", post_attn),
        ("block_0_post_conv", post_conv),
        ("block_0_post_ff2",  post_ff2),
        ("block_0_shadow",    block_0_out),
    ]:
        np.save(out / f"{name}.npy", t[0].detach().cpu().numpy().astype(np.float32))
        print(f"[dump-sub] {name}: {tuple(t[0].shape)}")


if __name__ == "__main__":
    main()
