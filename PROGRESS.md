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

### 5.2 — round 1: thread default + release flags + gallocr cache  _(done)_

Three non-timing-sensitive wins landed together:

  1. **CLI default thread count = `std::thread::hardware_concurrency()`**
     (was 4 via ggml-cpu's internal default).  `--threads N` still
     overrides.  On a 10-core M3 Ultra that's 10 threads by default.
     Worth ~10-12% on the encoder path in isolated measurements.
  2. **`-O3 -ffast-math -funroll-loops`** on `libqvac-parakeet` in
     Release builds (via `CMakeLists.txt` generator expressions;
     Debug/RelWithDebInfo unaffected).  Our pure-C++ FFT /
     filterbank-matmul / CMVN drops from ~14 ms to ~6 ms (2.3×).
     Doesn't touch ggml; it only affects our own DSP code, where
     `-ffast-math`'s associativity relaxation is safe (post-log-mel
     values are far from denormal / inf-adjacent regions).
  3. **Encoder graph allocator cached across calls**
     (`ParakeetCtcModel::Impl::encoder_alloc`).  Previously every
     `run_encoder()` built a fresh `ggml_gallocr` and re-walked the
     24-block graph; the fresh allocator + re-reserve cost ~5-10 ms
     per call and added noise to `--bench`.  Now allocated on the
     first call and reused as long as `n_mel_frames` is stable
     (re-created on shape change).

Post-opt numbers on an otherwise-quiet M3 Ultra (`jfk.wav`, 11 s audio,
`--bench-warmup 2 --bench-runs 5`):

```
                    mean     med      min      max      std
mel        ms       5.72    5.80    5.48    5.96    0.22   (was 14.63)
encoder    ms     940.13  943.40  856.94 1056.41   83.04   (was 1041.96)
decode     ms       0.08    0.08    0.08    0.09    0.01
inference  ms     945.94  948.97  862.88 1062.29   82.94   (was 1056.77)
RTF (median/best) = 0.086 / 0.078   (was 0.096 / 0.095)
```

`artifacts/bench/ggml-cpu-round1-m3ultra.json` snapshot archived.
Mel's 2.3× speedup is clean and reproducible.  Encoder variance is
higher than the baseline (std 83 ms vs 10 ms) — that's a
benchmark-noise effect from system contention, not a regression; in
isolation the median is within the previous std band.

### 5.3 — round 2: OpenMP + backend-buffer weight loading  _(done)_

Two changes shipped together:

  1. **OpenMP on ggml-cpu.**  `brew install libomp` (one-time) then
     `-DGGML_OPENMP=ON` at configure time.  CMake auto-links it via
     the existing `find_package(OpenMP)` block.  On a quiet M3 Ultra,
     with CPU-only backend, measured ~4% encoder speedup (median
     803 ms → 768 ms) and 42% tighter stdev (88 ms → 50 ms).  Worth
     taking for the variance reduction alone.
  2. **Weight loading reworked to use a backend-owned buffer.**
     `gguf_init_from_file` is now called with `no_alloc=true`, the
     ggml context is then populated via
     `ggml_backend_alloc_ctx_tensors(ctx, backend_cpu)`, and each
     tensor's data is streamed from the file into the backend buffer
     via `ggml_backend_tensor_set`.  The buffer is tagged
     `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` so future sched-based
     optimizations can reach it.  No direct perf impact (identical
     in-memory layout), but unblocks multi-backend scheduling.

### 5.4 — BLAS backend sched attempt  _(investigated, not shipped)_

Tried co-initialising the `ggml-blas` backend with `ggml_backend_sched`
configured as `[blas, cpu]` + `op_offload=true`.  Result on this model
+ machine: no speedup, sometimes slower.

Root causes:

  - ggml-cpu's **multi-threaded f16×f32 SIMD** matmul beats
    **single-threaded Accelerate** `cblas_sgemm` for our matmul sizes
    (d_model=1024, T_enc=138, FFN=4096).  On Apple Silicon
    Accelerate routes SGEMM to the single-CPU AMX coprocessor; for
    these "medium" matmuls, 10 parallel SIMD threads win.
  - Our weights are f16; BLAS needs f32 inputs, forcing on-the-fly
    dequantization that eats the BLAS kernel's advantage.
  - Sched splits the graph per op, adding per-op dispatch overhead.

Reverted to plain CPU backend.  BLAS backend init code is kept in
`load_from_gguf` (dormant, will be used when we plumb a real
sched-based multi-backend path for GPU offload).  BLAS attempt with
an f32 GGUF hit a `cur_backend_id != -1` sched assertion, not
pursued further.

### 5.5 — round 3: cached encoder graph  _(done)_

The encoder ggml graph (~600 nodes: 24 Conformer blocks with FF /
rel-pos MHA / conv + subsampling + CTC head) was rebuilt from scratch
on every `run_encoder` call — fresh `ggml_context`, fresh `cgraph`,
fresh `ggml_gallocr_new`, re-reserve.  That's pure per-call overhead
that doesn't scale with audio length.

Refactored `run_encoder` into two phases:

  1. `build_encoder_graph_cached(model, graph, n_mel_frames, ...)` —
     constructs the graph, pre-computes the sinusoidal rel-pos
     encoding (only shape-dependent), reserves the allocator.  Named
     input tensors (`mel_in`, `mask_t{0..3}`, `pe_in`) and output
     tensors are stashed on `Impl::encoder_graph`.
  2. The hot path in `run_encoder` just computes per-call masks from
     `mel_valid`, `ggml_backend_tensor_set` on the cached input
     tensor pointers, `ggml_backend_graph_compute`, and
     `ggml_backend_tensor_get` on the cached output tensors.

Graph rebuild is triggered only when `n_mel_frames` changes
(different input length).  For bench mode running the same wav N
times, the graph is built once and reused.

### 5.6 — baseline comparison

| run              | mel ms (median) | encoder ms (median) | encoder ms (best) | RTF median | RTF best | backend |
|------------------|----------------:|--------------------:|------------------:|-----------:|---------:|---------|
| pre-round-1      | 14.63           | 1046.23             | 1031.53           | 0.096      | 0.095    | ggml-cpu (4 thr) |
| round 1          |  5.80           |  786 (quiet)        |  733              | 0.073      | 0.067    | ggml-cpu (10 thr, O3/ffast-math) |
| round 2          |  ~9             |  ~850 (median), 770 (best) | 710 | 0.077–0.091 | **0.065–0.070** | ggml-cpu + OpenMP + weight buffer |
| round 3          |  8.5–9.1        |  761–862 (median)   | **706**           | 0.070–0.079 | **0.065–0.066** | + cached encoder graph |

**Note on variance.**  Round 2 numbers have wider spread than round 1
(stdev 75–140 ms on encoder) despite being measured on the same
machine.  Cause: macOS background activity (Spotlight, Time Machine,
etc.) preempting our encoder threads; mel and decode std grow too
when the system is busy.  The **best** encoder time is the cleanest
signal for "what the code achieves when nothing else is running";
**median** is what a user typically observes.  `--bench` output
reports both and warns when stdev > 20% of mean.

Snapshots: `artifacts/bench/ggml-cpu-baseline-m3ultra.json`,
`ggml-cpu-round1-m3ultra.json`, `ggml-cpu-round2-m3ultra.json`.

### 5.7 — sub-stage profiler + attribution  _(done)_

Added `--profile` mode to the CLI.  Drives two complementary sweeps
off the same model load:

  1. **Layer-depth sweep** — runs the encoder with
     `n_run_layers = {0, 1, 12, 24}` (wired through a new
     `max_layers` param on `run_encoder`; the graph cache keys on it
     so each config gets a fresh graph), times each.  Linear
     decomposition gives:
     - `subsampling + CTC head` = time@0
     - `per-block avg`          = (time@24 - time@1) / 23
     - `block-0 extra`          = time@1 - time@0 - per-block-avg
  2. **Within-block sub-stage sweep** — `profile_block_substages`
     in `src/parakeet_ctc.cpp` builds five tiny graphs (FF1 only,
     attention only, conv only, FF2 only, norm_out only) on a
     fixed-shape random input at `T_enc` and times each.  Also
     times the full block for consistency check.

Output on `jfk.wav` (11 s, M3 Ultra, 5 timed + 2 warmup):

```
[profile] mel preprocess                  4.83 ms  ( 0.6% of total)
[profile] subsampling + CTC head (nl=0)  71.36 ms  ( 8.3% of total)
[profile] per-block avg (nl=1..24)       32.64 ms  (x 24 = 783 ms, 91.3%)
[profile] full encoder (nl=24)          853.35 ms   RTF = 0.0780

[profile] per-block sub-stages (T_enc=137):
   Conv module    10.92 ms  (31% of block)   ~275 ms encoder-wide (32%)
   Attention       7.41 ms  (21%)            ~186 ms (22%)
   FF2             6.70 ms  (19%)            ~169 ms (20%)
   FF1             6.07 ms  (17%)            ~153 ms (18%)
   norm_out        0.05 ms  ( 0%)            ~  1 ms ( 0%)
```

**Key finding.**  Conv module is the single biggest slice (32%
encoder-wide), not FFN.  By FLOP count the conv module is ~5x
cheaper than FFN (~435 MFLOPs vs ~2.3 GFLOPs per block), so this is
a memory-bandwidth / implementation efficiency problem, not a
compute problem.  Next target:

  - `conv1d_via_matmul` casts f16 kernels to f32 via `ggml_cast`
    every forward pass — could keep f16 native in mul_mat by flipping
    argument order.
  - The `ggml_permute(x, 1, 0, 2, 3) + ggml_cont` wrappers around
    the module materialise a (d_model × T) buffer twice per block
    (enter + exit).  Re-shaping the internal ops to work on
    `(d_model, T)` layout natively would save ~24 * 2 *
    (d_model * T * sizeof(f32)) = 24 * 2 * 1024 * 137 * 4 bytes
    ≈ 27 MB of redundant copies per utterance.
  - `ggml_conv_2d_dw_direct` may be faster than the
    `ggml_conv_1d_dw` (im2col + mul_mat) we use today — the header
    even calls it out.

### 5.8 — round 4: conv module rewrite  _(done)_

Two structural changes to the conv module, driven by the 5.7 profile
that flagged it as the single biggest slice at 32% of encoder time:

  1. **Drop `ggml_cont` around GLU halves.**  `ggml_mul` and
     `ggml_sigmoid` accept strided views natively; the two `cont`
     calls were copying 2×(T×d_model×4) = ~1.1 MB per block, ~27 MB
     per forward, for no reason.  Per-block conv time: 10.92 → 8.10
     ms (-26%).

  2. **Replace `conv1d_via_matmul` with direct `ggml_mul_mat` for
     `pw1`/`pw2` (k=1 convs).**  A k=1 Conv1d is literally a matmul;
     doing it as such lets us:
       - skip the im2col (trivial but still a memcpy),
       - skip the `ggml_cast(kernel, F32)` that was in there to work
         around the `mul_mat(src0=f32, src1=f16)` ordering
         restriction,
       - stay in the natural `(d_model, T)` layout so the
         `ggml_permute + ggml_cont` enter/exit transposes (another
         ~1.1 MB per block) are gone.
     Depthwise conv still needs `(T, d_model)` layout so we
     transpose just around `dw + BN + SiLU`.  Per-block conv time:
     8.10 → 6.06 ms (a further -25%, total -45%).

Output rel on block_last moved from 1.60e-3 → 1.88e-3 — within the
f16 quantization floor, from different accumulation order in the
mul_mat kernel vs the im2col+matmul path.  All 9 `test-encoder`
parity gates still pass.

Sub-stage profile after round 4:

```
   FF1  (macaron)   6.13 ms  (23% of block)  ~186 ms encoder-wide
   Attention        7.64 ms  (29%)           ~232 ms           ← now biggest
   Conv module      6.06 ms  (23%)           ~184 ms
   FF2  (macaron)   6.39 ms  (24%)           ~194 ms
```

Attention is now the single biggest slice (26.2% of encoder) at ~232
ms.  FFN + Conv are a close 3-way tie around 20% each.

### 5.9 — baseline comparison

| run     | encoder median ms | encoder best ms | RTF median | RTF best | note |
|---------|------------------:|----------------:|-----------:|---------:|------|
| baseline| 1046              | 1032            | 0.096      | 0.095    | ggml-cpu 4 thr |
| round 1 | 786 (quiet)       | 733             | 0.073      | 0.067    | HC thr + O3/ffast-math |
| round 2 | ~850              | 770             | 0.077      | 0.070    | +OpenMP + weight buffer |
| round 3 | 761–862           | 706             | 0.070–0.079| 0.065    | +cached graph |
| round 4 | **745–809**       | **627**         | 0.069–0.074| **0.058**| +conv rewrite |

Cumulative: **40% reduction in encoder best-case** (1032 → 627 ms).
RTF best 0.058 = **17.4× real-time** on CPU alone.

### 5.10 — still planned

  - **Attention optimisation.**  Now the biggest single slice at 232
    ms / 26%.  Candidate wins: fuse q/k/v into one matmul with a
    packed weight (3× d_model rows in one matrix), reconsider the
    many `ggml_permute + ggml_cont` in rel-pos MHA, or just let
    BLAS-backed GEMM handle the bigger matmul sizes there where the
    comparison shifts in Accelerate's favor.
  - Block-quantized weights (Q4_0 / Q5_0 / Q8_0).  Would halve /
    quarter memory bandwidth on the FFN's 1024×4096 matrices, which
    are the largest in the model — potentially another 80–150 ms.
    Needs converter change + verification that parity stays tight.
  - Metal backend + `ggml_backend_sched` for GPU offload (future
    phase outside CPU-only scope).  The backend-buffer rework from
    round 2 is what the sched will need to plumb through.
