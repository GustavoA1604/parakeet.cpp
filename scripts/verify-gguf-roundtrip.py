#!/usr/bin/env python3
"""Verify every GGUF tensor numerically matches the original NeMo state_dict.

For each GGUF tensor we:

  1. Recreate the expected NumPy array from the NeMo state_dict (applying
     the same layout transforms as scripts/convert-parakeet-ctc-to-gguf.py:
     squeezing the CTC Conv1d kernel, fusing conv-module BatchNorm into
     scale+shift).
  2. Read the GGUF tensor back through the gguf Python reader.
  3. Compare values.  For f32 tensors we require max-abs == 0 (bit-exact
     round trip).  For f16 tensors we assert the max-abs diff is within
     half-precision quantization, i.e. <= max(|w|) * 2**-10.
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import gguf


def fuse_bn(weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def load_sd(nemo_path: Path):
    with tarfile.open(nemo_path, 'r') as t:
        buf = io.BytesIO(t.extractfile('./model_weights.ckpt').read())
    return torch.load(buf, map_location='cpu', weights_only=True)


def build_expected(sd: dict, n_layers: int):
    def np32(t): return t.detach().cpu().numpy().astype(np.float32, copy=False)

    out = {
        'preproc.mel_filterbank': np32(sd['preprocessor.featurizer.fb'][0]),
        'preproc.window':         np32(sd['preprocessor.featurizer.window']),

        'encoder.subsampling.conv0.weight':    np32(sd['encoder.pre_encode.conv.0.weight']),
        'encoder.subsampling.conv0.bias':      np32(sd['encoder.pre_encode.conv.0.bias']),
        'encoder.subsampling.conv1_dw.weight': np32(sd['encoder.pre_encode.conv.2.weight']),
        'encoder.subsampling.conv1_dw.bias':   np32(sd['encoder.pre_encode.conv.2.bias']),
        'encoder.subsampling.conv1_pw.weight': np32(sd['encoder.pre_encode.conv.3.weight']),
        'encoder.subsampling.conv1_pw.bias':   np32(sd['encoder.pre_encode.conv.3.bias']),
        'encoder.subsampling.conv2_dw.weight': np32(sd['encoder.pre_encode.conv.5.weight']),
        'encoder.subsampling.conv2_dw.bias':   np32(sd['encoder.pre_encode.conv.5.bias']),
        'encoder.subsampling.conv2_pw.weight': np32(sd['encoder.pre_encode.conv.6.weight']),
        'encoder.subsampling.conv2_pw.bias':   np32(sd['encoder.pre_encode.conv.6.bias']),
        'encoder.subsampling.out.weight':      np32(sd['encoder.pre_encode.out.weight']),
        'encoder.subsampling.out.bias':        np32(sd['encoder.pre_encode.out.bias']),

        'ctc.decoder.weight': np32(sd['decoder.decoder_layers.0.weight'].squeeze(-1)),
        'ctc.decoder.bias':   np32(sd['decoder.decoder_layers.0.bias']),
    }

    for i in range(n_layers):
        k = f'encoder.layers.{i}'
        p = f'encoder.blk.{i}'

        out[f'{p}.norm_ff1.weight'] = np32(sd[f'{k}.norm_feed_forward1.weight'])
        out[f'{p}.norm_ff1.bias']   = np32(sd[f'{k}.norm_feed_forward1.bias'])
        out[f'{p}.ff1.linear1.weight'] = np32(sd[f'{k}.feed_forward1.linear1.weight'])
        out[f'{p}.ff1.linear1.bias']   = np32(sd[f'{k}.feed_forward1.linear1.bias'])
        out[f'{p}.ff1.linear2.weight'] = np32(sd[f'{k}.feed_forward1.linear2.weight'])
        out[f'{p}.ff1.linear2.bias']   = np32(sd[f'{k}.feed_forward1.linear2.bias'])

        out[f'{p}.norm_attn.weight'] = np32(sd[f'{k}.norm_self_att.weight'])
        out[f'{p}.norm_attn.bias']   = np32(sd[f'{k}.norm_self_att.bias'])
        out[f'{p}.attn.q.weight']    = np32(sd[f'{k}.self_attn.linear_q.weight'])
        out[f'{p}.attn.q.bias']      = np32(sd[f'{k}.self_attn.linear_q.bias'])
        out[f'{p}.attn.k.weight']    = np32(sd[f'{k}.self_attn.linear_k.weight'])
        out[f'{p}.attn.k.bias']      = np32(sd[f'{k}.self_attn.linear_k.bias'])
        out[f'{p}.attn.v.weight']    = np32(sd[f'{k}.self_attn.linear_v.weight'])
        out[f'{p}.attn.v.bias']      = np32(sd[f'{k}.self_attn.linear_v.bias'])
        out[f'{p}.attn.out.weight']  = np32(sd[f'{k}.self_attn.linear_out.weight'])
        out[f'{p}.attn.out.bias']    = np32(sd[f'{k}.self_attn.linear_out.bias'])
        out[f'{p}.attn.pos.weight']  = np32(sd[f'{k}.self_attn.linear_pos.weight'])
        out[f'{p}.attn.pos_bias_u']  = np32(sd[f'{k}.self_attn.pos_bias_u'])
        out[f'{p}.attn.pos_bias_v']  = np32(sd[f'{k}.self_attn.pos_bias_v'])

        out[f'{p}.norm_conv.weight'] = np32(sd[f'{k}.norm_conv.weight'])
        out[f'{p}.norm_conv.bias']   = np32(sd[f'{k}.norm_conv.bias'])
        out[f'{p}.conv.pw1.weight']  = np32(sd[f'{k}.conv.pointwise_conv1.weight'])
        out[f'{p}.conv.pw1.bias']    = np32(sd[f'{k}.conv.pointwise_conv1.bias'])
        out[f'{p}.conv.dw.weight']   = np32(sd[f'{k}.conv.depthwise_conv.weight'])
        out[f'{p}.conv.dw.bias']     = np32(sd[f'{k}.conv.depthwise_conv.bias'])

        bn_scale, bn_shift = fuse_bn(
            np32(sd[f'{k}.conv.batch_norm.weight']),
            np32(sd[f'{k}.conv.batch_norm.bias']),
            np32(sd[f'{k}.conv.batch_norm.running_mean']),
            np32(sd[f'{k}.conv.batch_norm.running_var']),
            eps=1e-5,
        )
        out[f'{p}.conv.bn.scale'] = bn_scale
        out[f'{p}.conv.bn.shift'] = bn_shift

        out[f'{p}.conv.pw2.weight'] = np32(sd[f'{k}.conv.pointwise_conv2.weight'])
        out[f'{p}.conv.pw2.bias']   = np32(sd[f'{k}.conv.pointwise_conv2.bias'])

        out[f'{p}.norm_ff2.weight'] = np32(sd[f'{k}.norm_feed_forward2.weight'])
        out[f'{p}.norm_ff2.bias']   = np32(sd[f'{k}.norm_feed_forward2.bias'])
        out[f'{p}.ff2.linear1.weight'] = np32(sd[f'{k}.feed_forward2.linear1.weight'])
        out[f'{p}.ff2.linear1.bias']   = np32(sd[f'{k}.feed_forward2.linear1.bias'])
        out[f'{p}.ff2.linear2.weight'] = np32(sd[f'{k}.feed_forward2.linear2.weight'])
        out[f'{p}.ff2.linear2.bias']   = np32(sd[f'{k}.feed_forward2.linear2.bias'])

        out[f'{p}.norm_out.weight'] = np32(sd[f'{k}.norm_out.weight'])
        out[f'{p}.norm_out.bias']   = np32(sd[f'{k}.norm_out.bias'])

    return out


def ggml_to_numpy(t):
    data = np.asarray(t.data)
    if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
        return data.view(np.float32).reshape(tuple(reversed(t.shape))).astype(np.float32)
    if t.tensor_type in (gguf.GGMLQuantizationType.F16,):
        return data.view(np.float16).reshape(tuple(reversed(t.shape))).astype(np.float32)
    raise RuntimeError(f"unexpected tensor type {t.tensor_type}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=Path("models/parakeet-ctc-0.6b.gguf"))
    ap.add_argument("--nemo", type=Path, default=Path("models/parakeet-ctc-0.6b.nemo"))
    args = ap.parse_args()

    print(f"[verify] reading {args.gguf}", file=sys.stderr)
    reader = gguf.GGUFReader(str(args.gguf))
    gg = {t.name: t for t in reader.tensors}
    print(f"[verify] loaded {len(gg)} tensors from GGUF", file=sys.stderr)

    arch = None
    n_layers = 0
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = bytes(field.parts[field.data[0]]).decode()
        elif field.name == "parakeet.encoder.n_layers":
            n_layers = int(field.parts[field.data[0]][0])
    print(f"[verify] arch={arch} n_layers={n_layers}", file=sys.stderr)

    print(f"[verify] loading {args.nemo}", file=sys.stderr)
    sd = load_sd(args.nemo)
    print(f"[verify] building expected tensor map", file=sys.stderr)
    expected = build_expected(sd, n_layers)

    missing = set(expected.keys()) - set(gg.keys())
    extra   = set(gg.keys()) - set(expected.keys())
    if missing:
        print(f"[verify] MISSING in GGUF ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"  - {k}")
        return 1
    if extra:
        print(f"[verify] UNEXPECTED extras in GGUF (ignoring): {sorted(extra)[:5]}")

    f32_fail = 0
    f16_worst_rel = 0.0
    f16_worst_name = ""
    f16_count = 0
    f32_count = 0
    for name, exp in expected.items():
        t = gg[name]
        got = ggml_to_numpy(t)
        if got.shape != exp.shape:
            print(f"[verify] SHAPE {name}: got {got.shape} expected {exp.shape}")
            f32_fail += 1
            continue
        diff = got.astype(np.float32) - exp.astype(np.float32)
        max_abs = float(np.abs(diff).max()) if diff.size else 0.0
        denom = float(np.abs(exp).max()) if exp.size else 0.0
        rel = max_abs / denom if denom > 0 else max_abs
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            f32_count += 1
            if max_abs != 0.0:
                f32_fail += 1
                print(f"[verify] F32 NOT bit-exact: {name}  max_abs={max_abs:.3e}")
        else:
            f16_count += 1
            if rel > f16_worst_rel:
                f16_worst_rel = rel
                f16_worst_name = name

    print(f"[verify] f32 tensors: {f32_count} bit-exact: {f32_count - f32_fail}/{f32_count}", file=sys.stderr)
    print(f"[verify] f16 tensors: {f16_count}  worst rel: {f16_worst_rel:.3e}  (worst: {f16_worst_name})", file=sys.stderr)
    f16_gate = 2 ** -10
    ok = (f32_fail == 0) and (f16_worst_rel < f16_gate)
    print(f"[verify] {'PASS' if ok else 'FAIL'}  (f16 gate: rel < 2^-10 = {f16_gate:.3e})", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
