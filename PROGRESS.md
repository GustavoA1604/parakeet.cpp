# qvac-parakeet.cpp — development journal

Chronological record of each bring-up and numerical-parity milestone,
mirroring `chatterbox.cpp/PROGRESS.md`'s methodology: every stage lands
with a per-stage `.npy` reference dumped from NeMo PyTorch and a C++
harness asserting rel error below a documented threshold.

## Phase 0 — scaffolding  _(done)_

- Added `CMakeLists.txt` modeled on `chatterbox.cpp` (`QVAC_PARAKEET_*`
  options, `QVAC_PARAKEET_USE_SYSTEM_GGML` escape hatch, install rules
  producing `qvac-parakeet::qvac-parakeet` so the eventual vcpkg port
  is a drop-in).
- Added `scripts/setup-ggml.sh` pinned to the same ggml commit
  (`58c38058`) chatterbox builds against.
- Vendored `dr_wav.h` + `npy.h` from `chatterbox.cpp/src/` for wav I/O
  and reference-tensor compare.
- Public headers under `include/qvac-parakeet/` expose
  `qvac_parakeet_cli_main`, `qvac_parakeet::ctc::Engine`, and the
  one-shot `transcribe_wav` API.
- CLI + library + test harnesses build green on macOS (arm64).

## Phase 1 — converter + GGUF round-trip  _(done)_

- `scripts/convert-parakeet-ctc-to-gguf.py` extracts `model_config.yaml`
  + `model_weights.ckpt` + `tokenizer.model` from the HF `.nemo`
  tarball and writes a single GGUF.
- Tensor naming is a flat namespace built for the C++ side:
  - `preproc.mel_filterbank` (80, 257)        — NeMo's `featurizer.fb`
  - `preproc.window` (400,)                   — NeMo's Hann symmetric window
  - `encoder.subsampling.{conv0,conv1_dw,conv1_pw,conv2_dw,conv2_pw,out}.{weight,bias}`
  - `encoder.blk.{i}.{norm_ff1,ff1.linear1,ff1.linear2,norm_attn,attn.{q,k,v,out,pos},attn.pos_bias_{u,v},norm_conv,conv.{pw1,dw,bn,pw2},norm_ff2,ff2.linear1,ff2.linear2,norm_out}.{weight,bias}`
  - `ctc.decoder.{weight,bias}`              — final Conv1d kernel_size=1, flattened to (vocab+1, d_model)
- Conformer conv-module BatchNorm is **fused at convert time** into
  (`scale`, `shift`) vectors so the C++ graph is BN-free.
- f16 default for 2-D projections / convs; f32 for biases / norms /
  BN-fused scale+shift / preprocessor buffers.
- Output: `models/parakeet-ctc-0.6b.gguf` (1.16 GiB f16).
- C++ `load_from_gguf` in `src/parakeet_ctc.cpp` loads every expected
  tensor, fills the typed `SubsamplingWeights` / `BlockWeights` /
  `CtcHeadWeights` structs, and rejects any missing tensor with a
  clear error.
- `qvac-parakeet --verbose` prints the full hyperparameter + tensor
  summary (verified against `model_config.yaml`).

## Phase 2 — mel preprocessor parity  _(done)_

- `scripts/dump-ctc-reference.py` drives NeMo PyTorch on a wav, emits
  `mel.npy`, `subsampling_out.npy`, `block_0_out.npy`,
  `block_last_out.npy`, `encoder_out.npy`, `logits.npy`,
  `greedy_ids.npy`, plus the text transcript.  For
  `test/samples/jfk.wav` (11 s), NeMo prints:
  > "and so my fellow americans ask not what your country can do for
  > you ask what you can do for your country".
- C++ `compute_log_mel`:
  - preemph y[t] = x[t] − 0.97·x[t−1] (x[0] pass-through, in-place reverse loop),
  - reflect-pad by `n_fft/2 = 256` (torch.stft `center=True, pad_mode='reflect'`
    convention),
  - 512-point radix-2 Cooley–Tukey complex FFT per frame, window placed
    symmetrically (zero-padded 56 on each side of the 400-sample Hann),
  - magnitude² → matmul against the GGUF filterbank,
  - `log(x + 2**−24)`,
  - per-feature (per-mel-bin) CMVN over `seq_len = ⌈n_samples/hop⌉`
    (sample std, `+ 1e-5`), with tail frames zeroed — matches NeMo's
    `normalize_batch('per_feature')`.
- `test-mel` on `jfk.wav`:
  ```
  c++ mel: (80, 1101)   ref mel: (80, 1101)
  rel = 1.656e-03   max_abs = 3.385e-01   (target: rel < 5e-3)
    inner (excluding last 2 frames):  rel = 1.116e-04   max_abs = 3.211e-03
  ```
  Inner-frame rel of 1.1e-4 is f32 FFT rounding noise (verified: error
  plateaus as soon as boundary frames are excluded).  Good enough — the
  encoder's first stage (subsampling + ReLU) is tolerant to this level of
  per-bin fluctuation.

## Phase 3a — Python shadow encoder  _(done)_

Before writing ~1500 LoC of ggml graph code, landed
`scripts/ref-encoder-from-gguf.py`: a pure-PyTorch FastConformer forward
that reads weights from our GGUF (via `gguf.GGUFReader`, not from the
NeMo state_dict).  Validates two things at once:

  1. GGUF tensor layout semantics (shapes, transposes, BN fuse, f16
     round-trip) match what the C++ side will read.
  2. Our understanding of NeMo's FastConformer-CTC forward is correct.

End-to-end on `test/samples/jfk.wav`:

```
[shadow] tensors=904  layers=24  d_model=1024  heads=8
[parity] subsampling_out     rel = 5.8e-04
[parity] block_0_out         rel = 5.0e-04
[parity] block_last_out      rel = 6.7e-04
[parity] encoder_out         rel = 6.7e-04
[parity] logits              rel = 2.1e-04
[shadow] transcript: and so my fellow americans ask not what your country can do for you ask what you can do for your country
[shadow] reference : and so my fellow americans ask not what your country can do for you ask what you can do for your country
[shadow] match     : True
```

All five stages at the f16 quantization floor.  Transcript is bit-equal
to NeMo.  The shadow is now the authoritative spec for the C++ port.

**Key debugging win along the way.** Initial shadow reported block_0 rel
~33% vs the stored `block_0_out.npy`.  Root cause: the original
`dump-ctc-reference.py` ran `model.transcribe()` before the hook-driven
forward, and `transcribe()` mutates the MHA module in place (in NeMo 2.7.2
it flips `use_pytorch_sdpa = True`), so the saved intermediate `.npy`s
reflected a post-transcribe state that differed from a cold forward by
~33% numerically — but was mathematically equivalent on greedy argmax,
hence produced the same transcript.  The saved refs are now captured
cold, with per-block outputs (`block_{0..23}_out.npy`) for finer C++
gates.

## Phase 3b — FastConformer encoder ggml graph  _(done)_

Ported the shadow line-by-line to ggml.  Full per-sub-stage parity on
`test/samples/jfk.wav`:

```
[test-encoder] stage B  subsampling_out        rel=1.156e-03  max_abs=3.661e+00  ok
[test-encoder] stage C0 post_ff1  (b0)         rel=9.970e-04  max_abs=1.074e+02  ok
[test-encoder] stage C1 post_attn (b0)         rel=9.984e-04  max_abs=1.073e+02  ok
[test-encoder] stage C2 post_conv (b0)         rel=9.987e-04  max_abs=1.073e+02  ok
[test-encoder] stage C3 post_ff2  (b0)         rel=1.000e-03  max_abs=1.072e+02  ok
[test-encoder] stage C  block_0_out            rel=1.060e-03  max_abs=8.134e-02  ok
[test-encoder] stage D  block_last_out         rel=1.602e-03  max_abs=2.481e-02  ok
[test-encoder] stage E  encoder_out            rel=1.602e-03  max_abs=2.481e-02  ok
[test-encoder] stage F  logits (log_softmax)   rel=1.359e-03  max_abs=1.933e-01  ok
```

Every stage at the f16 quantization floor.

Implementation (`src/parakeet_ctc.cpp`):

  - `subsampling_graph`: 5 convs (1 full + 2 dw/pw pairs) with the
    `MaskedConvSequential` time-mask propagation matching NeMo (mask
    applied before each conv + after each stride drop, lengths tracked
    via `calc_length`).
  - `compute_rel_pos_encoding`: host-side sinusoidal table of shape
    `(2T-1, d_model)`, positions from `T-1` down to `-(T-1)`; fed as a
    graph input tensor.
  - `conformer_block_graph`:
      * Macaron FF (LayerNorm + linear + SiLU + linear + 0.5 residual).
      * Rel-pos MHA: q/k/v/pos linears → reshape to
        `(HD, T, H)` / `(HD, 2T-1, H)` → two matmuls for AC/BD terms
        → Transformer-XL `rel_shift` via concat-zero-pad + reshape
        trick → softmax → matmul with V → output linear.  Identical
        topology to chatterbox's S3Gen attention block.
      * Conv module: pointwise(d → 2d) → **GLU split + sigmoid(half2)
        × half1** → depthwise k=9 → pre-fused BN → SiLU → pointwise
        d → d.  Pre-fused BN saves one op per block across 24 blocks.
      * Second Macaron FF.
      * Final LayerNorm out.
  - `run_encoder`: builds a single 24-block graph, allocates it with
    `ggml_gallocr`, marks per-stage capture tensors with
    `ggml_set_output` so gallocr doesn't reuse their buffers, uploads
    mel + 4 masks + pos_emb via `ggml_backend_tensor_set`, runs,
    extracts all captures.

Key debugging wins (caught in minutes thanks to the shadow):

  - Swapped `ggml_mul_mat` arg order in `conv1d_via_matmul` to avoid
    an `F32 × F16` assertion; kernels pre-cast to F32 when they're
    stored as F16 in the GGUF.
  - `ggml_set_output` on every capture tensor to survive graph
    compaction (before this, outputs past the first sub-stage were
    silently overwritten by downstream ops).
  - Used `ggml_sigmoid` (not `ggml_silu`) inside the conv module's
    GLU.  This was the one-line bug driving block_0 rel from ~1e-3 to
    ~5e-2; isolating via per-sub-stage captures (`block_0_post_ff1`,
    `block_0_post_attn`, `block_0_post_conv`, `block_0_post_ff2`) and
    comparing against shadow dumps pinned it on the first try.

## Phase 4 — CTC head + end-to-end C++ transcription  _(done)_

CTC linear is part of the encoder graph (final `ggml_mul_mat + bias`
on `encoder_out`).  `log_softmax` is computed host-side for
numerical stability (ggml lacks a stable `log_softmax` op, and
argmax doesn't need it).  Greedy decode + collapse-repeats +
strip-blank is a trivial CPU loop.  SentencePiece detokenize works
off the `tokenizer.ggml.tokens` string array (+ scores and piece
types) which the converter now emits alongside the raw proto bytes.

End-to-end on `test/samples/jfk.wav`:

```
$ ./build/qvac-parakeet --model models/parakeet-ctc-0.6b.gguf \
                       --wav   test/samples/jfk.wav --verbose
[BENCH] load=126.9ms mel=12.9ms enc=913.4ms dec=0.2ms total=1053.6ms tokens=26
and so my fellow americans ask not what your country can do for you ask what you can do for your country
```

Bit-equal to the NeMo reference transcript.  RTF ≈ 0.10 on Apple
Silicon CPU (11 s of audio transcribed in 1.05 s, ~10× faster than
real-time) on a single-core unoptimized build.

## Phase 5 — CPU optimization pass  _(in progress)_

### 5.0 — built-in benchmark harness  _(done)_

Added a `--bench` mode to the CLI so we can compare optimizations
accurately and reproducibly (same warm state, same repeat count, same
stats) without shelling out to `time`.

- `--bench`                       enable benchmark mode
- `--bench-runs N`                timed runs (default 3)
- `--bench-warmup N`              warmup runs, excluded from stats
                                  (default 2, absorbs the cold-cache +
                                  first-graph-allocator outlier)
- `--bench-json PATH`             dump structured JSON for comparing
                                  across runs or backends (ggml-cpu
                                  today, ggml-metal / onnxruntime later)

Per-stage stats include mean / median / min / max / stdev for mel,
encoder, decode, and total inference; the summary line highlights
`median` and `best` RTF (mean is reported too but gets noisy when a
warm run gets preempted by the OS).  Std > 20% of mean triggers a
visible warning so we don't silently chase variance.

### 5.1 — baseline (pre-optimization)

Machine: Apple M3 Ultra, macOS, single-core unoptimized Release build.
Model: `parakeet-ctc-0.6b.gguf` at f16 (1.16 GiB).  Threads: default
(`std::thread::hardware_concurrency()` via ggml-cpu).  Audio:
`test/samples/jfk.wav` — 11.00 s, 176 000 samples @ 16 kHz.
`--bench-warmup 2 --bench-runs 5`:

```
                    mean     med      min      max      std
mel        ms      14.63    14.65    14.11    15.10     0.42
encoder    ms    1041.96  1046.23  1031.53  1054.12    10.00
decode     ms       0.17     0.17     0.17     0.18     0.01
inference  ms    1056.77  1060.51  1046.02  1069.41    10.21
RTF (median/best) = 0.096 / 0.095    (realtime multiple = 10.4x / 10.5x)
model load         = 449 ms   (one-time, excluded from RTF)
```

- Encoder dominates inference (**98.6%** of wall time).
- Mel preprocessor is a ~1.4% slice (13–15 ms for 11 s of audio).
- Greedy decode + SentencePiece detokenize is effectively free (~0.17 ms).
- Std of 1% on inference across 5 warm runs → measurements are tight
  enough to catch ≥ 2% improvements without heroics.

JSON reference snapshot archived at
`artifacts/bench/ggml-cpu-baseline-m3ultra.json`.

### 5.2 — planned optimizations

  - OpenMP on the ggml build (`GGML_OPENMP=ON`) and thread-pool tuning
    on the cpu backend.
  - Accelerate framework BLAS linkage on macOS.
  - `-O3 -ffast-math -funroll-loops` for the qvac-parakeet code.
  - Per-utterance graph + gallocr cache (currently a fresh
    `ggml_context` + graph allocator is built per `run_encoder` call,
    adding ~10 ms of overhead and some noise on warm runs).
  - Larger-granularity profiling: split encoder time across
    subsampling / per-block / CTC head using `ggml_time_us()` hooks
    inside the graph.

Each optimization lands with a before/after row in the table and the
corresponding `artifacts/bench/*.json` snapshot committed alongside
the PROGRESS entry, so the impact is auditable.
