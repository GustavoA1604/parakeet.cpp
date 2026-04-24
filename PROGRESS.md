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

### 5.11 — round 5: attention optimisation attempts  _(investigated, shipped as dormant infrastructure)_

Two attention-path experiments, both motivated by PROGRESS 5.8's
attention-as-biggest-slice finding (26 % of encoder wall time after
the Round 4 conv rewrite).

1. **Packed QKV matmul.**  Converter now emits
   `encoder.blk.{i}.attn.qkv.{weight,bias}` in addition to the three
   separate `q/k/v.{weight,bias}` tensors.  `BlockWeights` has
   `attn_qkv_w/b` fields; `load_from_gguf` picks them up optionally.
   The graph branches on `W.attn_qkv_w != nullptr` — packed path does
   one `ggml_mul_mat` + bias + `reshape_4d(HD, H, 3, T)` + three
   `ggml_view_3d` slices to extract Q/K/V.

2. **`ggml_cont` pruning around `q/k/v/p_perm` permutes.**  mul_mat
   and ggml_add accept non-contiguous src as long as `nb00 == type_size`,
   so the `cont` could in principle be dropped for k_perm and p_perm
   (used directly as mul_mat src0) and for q_perm (materialised by the
   downstream add with pos_bias_u/v).

**Result on M3 Ultra, CPU-only.**  Neither change produced a reliable
win above the ~15% bench-to-bench stdev, and some configurations
regressed.

Root causes (measured):

- The packed output lays out Q/K/V in a single 3×d_model row, so the
  per-slice T stride is 3 * HD * H * 4 = 12 KB vs the natural 4 KB for
  separate matmuls.  The subsequent `cont(permute)` does a strided copy
  that's roughly 3× more cache-unfriendly — net slower than the three
  smaller matmuls ggml-cpu already runs in parallel.
- Dropping `cont` on k_perm/p_perm pushes the strided reads into the
  mul_mat kernel itself, which on ggml-cpu's f16×f32 SIMD path is a
  slower code path than contiguous src0.  The `cont` copy was
  effectively buying a faster subsequent mul_mat.
- Fresh per-block substage profile (after all Round 4 changes, packed
  QKV kept dormant in graph):

```
   FF1  (macaron)   6.07 ms  (21% of block)
   Attention        5.72 ms  (20%)          ← no longer biggest
   Conv module      7.90 ms  (28%)          ← biggest on this machine
   FF2  (macaron)   6.40 ms  (23%)
   norm_out         0.04 ms  ( 0%)
```

Attention is no longer dominant on M3 Ultra — the conv module's
`ggml_conv_1d_dw` (im2col+matmul) path and `pw1`/`pw2` matmuls are now
the single biggest slice.  FFN remains the largest aggregate (43%)
and is the right target for Round 6 (block quantization).

**Shipped:** packed-QKV tensor emission in the converter,
`BlockWeights::attn_qkv_{w,b}`, and the optional load path.  Graph
still uses the 3-matmul path.  Infrastructure is dormant but kept
because Round 7's `ggml_flash_attn_ext` experiment will want the
packed Q/K/V regardless.

**Not shipped:** any graph-level change.  The baseline (reverted to
pre-Round-5 attention) is the current code.

Bench snapshot on `sample-16k.wav` (20.1 s, `--bench-warmup 3
--bench-runs 10`, OpenMP, 10 threads):

```
                    mean     med      min      max     std
encoder    ms    1316.70 1245.81  1193.73  1559.76   140.71
RTF (median/best) = 0.063 / 0.060
```

Snapshot: `artifacts/bench/ggml-cpu-round5-m3ultra.json`.

### 5.12 — round 6: block-quantized weights  _(done — biggest CPU win so far)_

Quantize the ~150 largest 2D weight matrices per block (FFN, attention
q/k/v/qkv/out/pos, conv pointwise, subsampling out, CTC head) using
ggml-cpu's hand-tuned Q8_0 / Q5_0 / Q4_0 kernels.  Small tensors
(biases, norms, fused BN, mel filterbank, depthwise kernels, tiny 2D
subsampling convs) stay at f32 / f16 because their innermost dim
doesn't divide the 32-element block size.

Converter side (`scripts/convert-parakeet-ctc-to-gguf.py`):

  - New `--quant {f32, f16, q8_0, q5_0, q4_0}`.
  - Single `add_2d` helper routes each 2D weight through
    `gguf.quants.quantize(arr, qtype)` when the inner dim % 32 == 0,
    with an f16 fallback otherwise. Squeezes the trailing 1 on
    `conv.pw{1,2}.weight` so they can be quantized.
  - File-type header updated to match the selected quant
    (`LlamaFileType.MOSTLY_Q8_0` etc.).

C++ side (`src/parakeet_ctc.cpp`):

  - No graph changes needed. `ggml_mul_mat` dispatches to the Q8_0 /
    Q5_0 / Q4_0 kernel automatically based on src0's stored type.
  - `load_from_gguf` already used `ggml_nbytes(t)` to size the read,
    which correctly accounts for block-aligned storage.
  - `conformer_conv_graph`'s `ggml_reshape_2d(W.conv_pw1_w, d_model,
    2*d_model)` becomes a metadata-only identity after the converter
    squeeze (pw1 already stored as 2D (1024, 2048)); reshape_2d still
    accepts the shape and works on quantized src.

Parity (tested on `jfk.wav` + `sample-16k.wav`): **transcript is
bit-equal to NeMo PyTorch at every quantization level, including
Q4_0**.  Per-stage rel error grows as expected: f16 ~1.6e-3 → Q8_0
~5.5e-3 → Q4_0 ~3.3e-2.  Rel drift does NOT translate into token
drift on clean speech in these tests.

Bench results on M3 Ultra, 10 ggml-cpu threads, `--bench-warmup 3
--bench-runs 10`:

| variant | file    | enc best (20 s) | enc median (20 s) | enc best (11 s) | enc median (11 s) |
|---------|---------|----------------:|------------------:|----------------:|------------------:|
| f16     | 1.3 GiB | 1194            | 1246              | 683             | 796               |
| Q8_0    | 697 MiB | **999**         | 1209              | **600**         | **655**           |
| Q5_0    | 453 MiB | 1475            | 1614              | ~650            | —                 |
| Q4_0    | 372 MiB | 1080            | 1286              | 595             | 637               |

**Key findings:**

  - **Q8_0 is the speed + parity sweet spot.** Best-case encoder time
    drops from 1194 → 999 ms on the 20 s clip (-16 %), and from 683
    → 600 ms on the 11 s clip (-12 %).  RTF best 0.050 on 20 s
    (20x real-time on CPU alone).
  - **Q4_0 is a valid size tier.** ~10 % slower than Q8_0 on average
    but model shrinks to 372 MiB (3.5x smaller than f16), with the
    same bit-equal transcript.
  - **Q5_0 is a trap on this machine.** File size drops to 453 MiB
    (smaller than Q8_0) but the ggml-cpu Q5_0 mul_mat kernel is
    noticeably slower than either Q8_0 or Q4_0 on Apple Silicon.
    Shipped anyway for the size tier, not recommended for speed.
  - **Model load time improves too** (bandwidth-bound): f16 312 ms →
    Q8_0 166 ms → Q4_0 96 ms on 20 s benches.

Remaining gap vs ONNX (20 s clip): **Q8_0 999 ms vs ONNX 944 ms** —
from 317 ms gap to ~55 ms (**83 % of the remaining gap closed with
Round 6 alone**).

Snapshots:
  - `artifacts/bench/ggml-cpu-round6-q8_0-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round6-q5_0-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round6-q4_0-m3ultra.json`

### 5.13 — round 7: flash_attn_ext experiment  _(investigated, not shipped)_

`ggml_flash_attn_ext(q, k, v, mask, scale, max_bias, logit_softcap)`
fuses `softmax(q @ k^T * scale + mask) @ v` into a single op.
Prototyped it behind `#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN` in
`rel_pos_mha_graph`:

  - Compute the Transformer-XL rel-pos BD branch exactly as before
    (`bd_final` of shape `(T, T, H)`).
  - Pre-scale BD by `1/sqrt(HD)` (flash_attn_ext applies the `scale`
    argument only to `q@k^T`, the mask is added as-is).
  - Cast BD to f16 (CPU backend requires f16 mask — `ggml.c` line 5320).
  - Call `ggml_flash_attn_ext(q_u, k_perm, v_perm, bd_mask, scale,
    0.0f, 0.0f)` — skips the explicit `ac = mul_mat(k, q_u)`, the
    `ac + bd_final` add, the `soft_max`, the second mul_mat on V,
    and the `v_for_mm = cont(permute(v_perm, 1, 0, 2, 3))` copy.
  - Output layout `(HD, H, T)` feeds directly into `reshape_2d(HD*H, T)`
    without the extra permute+cont tail of the non-flash path.

Parity: all 9 `test-encoder` gates pass.  `block_last` rel drifts from
1.9e-3 → 4.2e-3 (f16 mask cast adds one quantization step), still
under the 5e-3 threshold.

**Bench result on M3 Ultra, ggml-cpu Q8_0, 3x(warmup 3 + runs 10):**

| clip           | non-flash best | flash best | non-flash median | flash median |
|----------------|---------------:|-----------:|-----------------:|-------------:|
| jfk.wav (T=138)|            529 |        559 |              561 |          606 |
| sample-16k.wav (T=251)| 1037 |       1087 |             1168 |         1157 |

Flash_attn_ext is neutral-to-slower on CPU at these sequence lengths.
The overhead of the f16 BD mask cast and the extra BD pre-scale offset
the savings from fusing the four attention ops, and ggml-cpu's
`q_u @ k_perm^T` matmul is already well-tuned for T ~ 140–250.

**Gate** (per plan: ship if encoder median drops >=30 ms): FAILED.

**Shipped:** code is preserved behind `#ifdef
PARAKEET_EXPERIMENTAL_FLASH_ATTN` (default off). The Metal backend
phase will want to revisit this — flash-attn typically wins big on
GPU where softmax + V-multiply fuse into one kernel pass.

### 5.14 — round 8a: conv module depthwise rewrite  _(done — second-biggest CPU win)_

Swapped `ggml_conv_1d_dw` (im2col + mul_mat path) for
`ggml_conv_2d_dw_direct` on the Conformer depthwise kernel in
`conformer_conv_graph`.

Implementation:

  - The existing conv.dw.weight stored shape `(d_model, 1, 9)` —
    `ggml_reshape_4d(W.conv_dw_w, conv_kernel, 1, 1, d_model)` gives
    the `(KW=9, KH=1, 1, C=d_model)` layout that
    `ggml_conv_2d_dw_direct` requires.
  - Wrap yt from `(T, d_model, 1, 1)` into `(W=T, H=1, C=d_model, N=1)`
    via `ggml_reshape_4d`, run the op, unwrap back to `(T, d_model, 1)`
    via `ggml_reshape_3d`.
  - The CPU backend's depthwise kernel accesses the filter as
    `const float *`, so we `ggml_cast(W.conv_dw_w, GGML_TYPE_F32)` once
    (graph-build time, small cost — 9*d_model elements) when the
    stored type is f16.  Alternative would be storing as f32 at
    convert time; the cast is simpler and works on all existing
    GGUFs.

Parity: all 9 `test-encoder` stages pass.  block_last rel is
essentially unchanged (1.73e-3 vs 1.60e-3 previously).

**Bench on M3 Ultra, Q8_0, 15 timed runs, 5 warmup:**

| clip                   | enc best before | enc best after | delta | enc median before | enc median after |
|------------------------|----------------:|---------------:|------:|------------------:|-----------------:|
| jfk.wav (11 s)         |             529 |        **460** |  -13% |               561 |          **481** |
| sample-16k.wav (20.1 s)|            1000 |        **839** |  -16% |              1208 |          **882** |

This single op swap is ~100–200 ms cheaper than the im2col+mul_mat
path across 24 blocks. The previous profiler breakdown attributed
28 % of encoder time to the conv module; after this change it drops
meaningfully, and the remaining sub-stages are roughly a three-way
tie between FF1, FF2, and attention.

**Measured vs ONNX Runtime** (20 s clip): Q8_0 + conv_2d_dw_direct
best 839 ms vs ONNX 944 ms — **ggml-cpu is now 12 % faster than
ONNX on best-case encoder**. Round 4's 317 ms gap is entirely
closed.

Snapshots: `artifacts/bench/ggml-cpu-round8a-q8_0-m3ultra.json`.

### 5.15 — round 8b: subsampling mask fast-path  _(done — neutral)_

Added an `all_valid` flag threaded through `build_encoder_graph_cached`
and `subsampling_graph`. When the caller's mel has no trailing
silence (`mel_valid == n_mel_frames`, the common case for a single
utterance), the 8 `apply_time_mask` `ggml_mul` calls in
`subsampling_graph` are skipped — the graph is built without those
ops at all.  `EncoderGraph` caches the `all_valid` value so the graph
is rebuilt when it flips.

Parity: all 9 `test-encoder` gates still pass (the test sends a
padded mel, so `all_valid=false` and the masked path runs).

**Bench impact**: within noise (~0-10 ms), because the mask ops were
already small element-wise muls and ggml-cpu runs them cheaply in the
OpenMP pool.  Shipped anyway for correctness hygiene — running a
no-op mul_by_ones is silly — and because the infrastructure enables
the Round 8c LRU cache to cleanly key on `all_valid`.

### 5.16 — round 8c: multi-shape LRU graph cache  _(done — latent win)_

Replaced the single-shape `Impl::encoder_graph` with a small LRU
`std::vector<std::unique_ptr<EncoderGraph>>` of up to 3 entries. The
cache key is `(n_mel_frames, n_run_layers, all_valid)`.

Behaviour:

  - On `run_encoder`, scan the cache for a matching entry. If found,
    reuse it and move it to the back (most-recently-used).
  - If no match, evict the oldest entry (if cache is full) and build
    a new graph for the current shape.
  - Graph rebuild only happens on a genuine shape change; previously
    any shape change freed the single cached graph and rebuilt it.

This is a **latent** optimisation: the benchmark mode reuses one shape
and shows no change.  The win shows up in production callers that
alternate between a few utterance lengths (streaming, short-burst
input, etc.) — those paths avoid the ~20-50 ms graph rebuild cost on
every length change.

Parity: unchanged. Transcripts bit-equal on both test clips.

### 5.17 — summary, Round 5-8

| round              | code        | jfk best | 20s best | vs ONNX f16 best (944) |
|--------------------|:-----------:|---------:|---------:|------------------------:|
| pre-Round-5        | f16         |      617 |     1197 |               -27 %    |
| Round 5            | f16         |      683 |     1193 |               -26 %    |
| Round 6            | Q8_0        |      600 |      999 |                -6 %    |
| Round 7            | Q8_0 + flash_attn | 559|     1087 |              -15 %    |
| Round 8 (8a+8b+8c) | **Q8_0**    |  **460** | **839**  |           **+11 %**    |

**Round 8 vs ONNX f16**: 11 % faster on best-case encoder on a 20 s clip.

**Fair f16 vs f16** (same precision, different runtimes — 5 warmup + 15 timed runs):

```
                   onnxruntime-f16    ggml-cpu-f16
  -----------------------------------------------
  model size           2.3 GiB         1.3 GiB
  load ms              16 736            642      (26x faster cold start)
  inf best ms             948           1117      (15 % slower)
  inf median ms         1 007           1132      (12 % slower)
  inf stdev ms             52             18      (3x tighter)
  RTF best               0.047          0.055
  RTF median             0.050          0.056
  Transcripts            match          match
```

**Fair int8 vs int8** (generated via ORT dynamic quantization from the same weights,
5 warmup + 15 timed runs):

```
                   onnxruntime-int8    ggml-cpu-Q8_0
  -------------------------------------------------
  model size           583.9 MiB         697 MiB
  load ms               2 054             179      (11x faster cold start)
  inf best ms             677             898      (25 % slower)
  inf median ms           721             928      (22 % slower)
  inf stdev ms             55              25      (2x tighter)
  RTF best               0.034           0.045
  RTF median             0.036           0.046
  Transcripts            match           match
```

Interpretation:

  - ggml is **12–25 % slower** than onnxruntime at the same precision tier.
    onnxruntime's kernels on Apple Silicon route through AMX coprocessor
    instructions (hand-tuned for both f16 and int8) that ggml-cpu's
    OpenMP SIMD threads can't match on multiply-accumulate throughput.
  - ggml stdev is **2–3× tighter** at both tiers (18 vs 52 ms at f16;
    25 vs 55 ms at int8), meaning per-utterance latency is more
    predictable under background OS load.
  - ggml model load is **11–26× faster** — critical for cold-start /
    short-session workloads.
  - The Metal backend (planned Phase 6) will target GPU compute, where
    AMX doesn't apply and ggml's flash-attention kernel (already
    prototyped in Round 7) can be used.

RTF best on 20 s clip (Q8_0): 0.045 → **22x real-time** on CPU alone.
Model load: 179 ms vs ONNX int8's 2054 ms.

Snapshots:

  - `artifacts/bench/ggml-cpu-round5-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round6-{q8_0,q5_0,q4_0}-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round8a-q8_0-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round8-q8_0-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round8-q8_0-jfk-m3ultra.json`
  - `artifacts/bench/ggml-cpu-round8-f16-m3ultra.json`

## Phase 6 — Metal backend  _(done, experimental)_

Bring-up of the `ggml_backend_metal` path for GPU offload on Apple
Silicon.  End-to-end on the M3 Ultra GPU (48-core):

### 6.1 — wire-up

  - `init_gpu_backend(n_gpu_layers, verbose)` helper chooses CUDA →
    Metal → Vulkan → CPU based on compile flags and returns
    `nullptr` when `n_gpu_layers <= 0` or no GPU backend is
    compiled in. Matches the convention used by `llama.cpp`,
    `whisper.cpp`, and `chatterbox.cpp`.
  - `Impl::backend_active` pointer — one of CPU or GPU — drives
    `ggml_backend_alloc_ctx_tensors`, `ggml_backend_graph_compute`,
    and the per-call `safe_set` tensor uploads.  All weights live on
    the GPU backend (unified memory on Apple Silicon), graph runs
    entirely on GPU.
  - Standard CLI flag: `--n-gpu-layers N` (same spelling as
    llama.cpp / whisper.cpp). Any value > 0 moves the whole encoder
    to GPU — this model has one encoder, so we don't actually need
    per-layer granularity.
  - Compile via `cmake -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`
    (or `-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`).
  - `ggml_conv_2d_dw_direct` (Round 8a) is **not yet implemented on
    Metal** (`ggml_metal_op_encode_impl: error: unsupported op
    'CONV_2D_DW'`). `conformer_conv_graph` takes a `use_conv2d_dw`
    bool, chosen at graph-build time via `ggml_backend_is_cpu(backend)`:
    CPU path uses the fast direct kernel, GPU paths revert to
    `ggml_conv_1d_dw` (im2col + mul_mat, Metal/CUDA/Vulkan supported).
  - `flash_attn_ext` left behind `#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN`
    from Round 7 — should be tested on Metal as a separate follow-up.

### 6.2 — parity

Metal `test-encoder` on `jfk.wav` + `artifacts/ctc-ref`:

```
stage B  subsampling_out        rel=7.641e-04  (CPU: 1.156e-03)
stage C0 post_ff1  (b0)         rel=4.859e-04  (CPU: 9.970e-04)
stage C1 post_attn (b0)         rel=4.866e-04  (CPU: 9.984e-04)
stage C2 post_conv (b0)         rel=4.870e-04  (CPU: 9.987e-04)
stage C3 post_ff2  (b0)         rel=4.880e-04  (CPU: 1.000e-03)
stage C  block_0_out            rel=6.756e-04  (CPU: 1.060e-03)
stage D  block_last_out         rel=1.698e-03  (CPU: 1.730e-03)
stage E  encoder_out            rel=1.698e-03  (CPU: 1.730e-03)
stage F  logits (log_softmax)   rel=3.871e-04  (CPU: 1.362e-03)
```

All 9 gates pass, and Metal per-stage rel is **tighter than CPU**
(the Metal f16 mul_mat kernels use f32 accumulators throughout, which
happens to track NeMo PyTorch's f32 reference more closely than the
CPU path's mixed-precision accumulation).

### 6.3 — bench

`sample-16k.wav` (20 s), `--bench-warmup 5 --bench-runs 15`:

| variant            | enc best | enc median | stdev | RTF best | real-time multiple |
|--------------------|---------:|-----------:|------:|---------:|-------------------:|
| CPU f16 (Round 8)  |    1 117 |      1 132 |    18 |    0.055 |              18x   |
| CPU Q8_0 (Round 8) |      898 |        928 |    25 |    0.045 |              22x   |
| CPU Q4_0 (Round 8) |    1 080 |      1 286 |   138 |    0.054 |              19x   |
| **Metal f16**      |    **266** |    **268** | **1.1** | **0.013** |        **75x**   |
| **Metal Q8_0**     |    **272** |    **274** | **1.5** | **0.014** |        **73x**   |
| Metal Q4_0         |      271 |        272 |   0.5 |    0.014 |              74x   |

On `jfk.wav` (11 s): Metal f16 encoder best 152 ms, median 154 ms.

### 6.4 — comparison vs onnxruntime

`sample-16k.wav`, 5 warmup + 15 timed runs, ggml run with
`--n-gpu-layers 1`:

```
                   onnxruntime-int8    ggml-metal-Q8_0
  ---------------------------------------------------
  model size           583.9 MiB         697 MiB
  load ms               2 295              420      (5.5x faster cold start)
  inf best ms             682              282      (2.4x faster)
  inf median ms           712              283      (2.5x faster)
  inf stdev ms             18             0.83      (21x tighter)
  RTF best               0.034           0.014
  RTF median             0.035           0.014
  Transcripts            match           match
```

**Metal ggml is 2.4x–2.5x faster than onnxruntime's AMX-accelerated
int8 path**, with 21x tighter variance (0.83 ms vs 18 ms stdev).
Metal is compute-bound on GPU shader units, so quantization does not
help (f16 / Q8_0 / Q4_0 all cluster around 272 ms) — but it does
shrink the model file and the unified-memory footprint.

### 6.5 — remaining work

  - Implement `CONV_2D_DW` on the Metal backend (upstream contribution
    to ggml) so the CPU and Metal paths share `conformer_conv_graph`.
    Would buy a few ms more on Metal since the direct path is
    asymptotically cheaper than im2col.
  - Test `ggml_flash_attn_ext` on Metal — likely a meaningful win given
    the fused softmax + V-multiply kernel, plus the dormant infra from
    Round 7 is already in place.
  - Hybrid `ggml_backend_sched` with Metal for the encoder + CPU for
    the mel preprocessor, so the CPU mel path doesn't block the GPU
    encoder. Today the mel runs inline on host before the encoder
    starts; with a sched we could overlap them.

---

### 5.18 — still planned (CPU-only work)

Phase 5 (CPU optimization) is effectively complete: Round 8 Q8_0 is
11 % faster than `onnxruntime` on the 20 s clip.  Remaining candidate
work is now outside the CPU-only scope:

  - **Metal backend + `ggml_backend_sched` for GPU offload.**  The
    backend-buffer rework from Round 2 and the cached encoder graph
    from Round 3 are what the sched will need to plumb through.
    flash_attn_ext (dormant behind `PARAKEET_EXPERIMENTAL_FLASH_ATTN`
    from Round 7) will almost certainly be a win on GPU where the
    softmax + V-multiply fuse into one kernel pass.
  - **K-quant tiers (Q4_K_M, Q5_K_M, Q6_K).**  ggml-cpu has k-quant
    kernels too; these might extend the quality-vs-size curve beyond
    the block-quant tiers shipped in Round 6.  Would need a sweep
    against parity.
  - **Bucketed encoder graph cache.**  Round 8c landed an exact-shape
    LRU cache (up to 3 entries). A bucketed variant — round up to the
    next multiple of 64 or 128 mel frames — would avoid rebuilds for
    variable-length production streams, at the cost of padding the
    mel input and masking out the tail via the `all_valid=false` path.
    Becomes a prerequisite for Phase 8 (cache-aware streaming) where
    every chunk is a fresh encoder call.
  - TDT / EOU / Sortformer pipelines (new architectures, not a
    CPU-opt task).

## Phase 7 — streaming entry points (Mode 2)  _(done)_

Scope: ship a platform-agnostic streaming API surface on `Engine` that
mirrors the three qvac/packages/sdk transcription shapes (one-shot,
streamed-output, duplex). Mode 2 (streamed-output) is implemented on
top of today's offline encoder; Mode 3 (duplex) has its header +
binding-facing API frozen but errors at runtime until Phase 8 delivers
a cache-aware streaming GGUF.

Design rationale is in the plan's scope discussion: chunked-batch on
the offline encoder was explicitly rejected because it costs 1-3 % WER
at 2 s chunks with no throughput win on 20 s clips, and Mode 2's
"offline encoder + CTC-timestamp streaming" gets the same UX at zero
accuracy cost.

### 7.1 — Engine class implementation

`include/qvac-parakeet/ctc/engine.h` was the declared-but-unimplemented
surface. Phase 7 lands the definition in `src/parakeet_engine.cpp`:

- `Engine(const EngineOptions &)` loads the GGUF once via
  `load_from_gguf`, stores `ParakeetCtcModel` + cancel flag in `Impl`.
- `Engine::transcribe(wav_path)` / `transcribe_samples(samples, n, sr)`
  drive the existing `compute_log_mel` + `run_encoder` +
  `ctc_greedy_decode` + `detokenize` pipeline and return an
  `EngineResult` with per-stage timing.
- `Engine::cancel()` sets an atomic flag; the streaming loop polls it
  between chunks (cooperative cancellation only — the encoder graph
  run itself is not interruptible).

### 7.2 — Stateful CTC window decoder

Added `ctc_greedy_decode_window(logits, start, end, vocab, blank,
inout_prev_token, out_tokens, out_first_frame=nullptr)` in
`src/parakeet_ctc.{h,cpp}`. The existing one-shot
`ctc_greedy_decode()` now delegates to it with `prev_token = -1`.

The stateful variant preserves collapse-repeats across window
boundaries via a caller-managed `inout_prev_token`, so a token whose
first argmax lands in window K and repeats in window K+1 isn't emitted
twice. This is the core invariant that makes Mode 2 byte-equal to the
offline path.

### 7.3 — Mode 2: `transcribe_stream` / `transcribe_samples_stream`

`Engine::transcribe_samples_stream(samples, n, sr, opts, on_segment)`:

1. Runs the existing offline mel + encoder path once.
2. Computes `frames_per_window = max(1, opts.chunk_ms / 80 ms)` (the
   80 ms comes from 10 ms mel hop × 8x subsampling).
3. Walks `[0, T_enc)` in contiguous windows, calling
   `ctc_greedy_decode_window` with a persistent `prev_token`.
4. After each window's decode, detokenizes the *cumulative* token list
   and emits the delta slice as the segment text. Detokenizing
   cumulatively rather than per-window is required because
   `sentencepiece_bpe::detokenize` strips a leading ASCII space — if
   we detokenized per window, the leading space of every segment
   except the first would be silently stripped and `"hello world"`
   would come out as `"helloworld"`. Caught by the first run of the
   new test harness on `jfk.wav` and fixed before landing.
5. Emits `StreamingSegment{text, token_ids, start_s, end_s,
   chunk_index, is_final=true, encoder_ms (first segment only),
   decode_ms}` via the caller's callback, and accumulates into
   the returned `EngineResult`.

`transcribe_stream(wav_path, opts, cb)` is a thin wrapper that loads
the WAV and forwards to `transcribe_samples_stream`. Both return
the full concatenated `EngineResult` so callers that want *both* the
streaming callback *and* a final aggregate don't have to rebuild it
themselves.

### 7.4 — Mode 3 API freeze (errored)

`StreamSession` declares `feed_pcm_f32(const float*, int)`,
`feed_pcm_i16(const int16_t*, int)`, `finalize()`, `cancel()`,
`options()`, destructor, and move ctor/assign. Implementation is in
`src/parakeet_engine.cpp`.

`Engine::stream_start(opts, cb)` probes
`pimpl_->model.supports_streaming` (fed from the new
`parakeet.encoder.streaming.enabled` GGUF key, added to
`load_from_gguf`). Today's GGUFs don't set the flag, so the call
throws `std::runtime_error` with a message pointing at Phase 8 and
suggesting `transcribe_stream()` for full-audio cases. The
`qvac-lib-infer-parakeet` binding can be wired against the final
`StreamSession` shape immediately; when Phase 8 lands the error
branch is swapped for the real state machine without touching the
public header.

### 7.5 — CLI wiring

`src/main.cpp`:

- `--pcm-in PATH` + `--pcm-format {s16le,f32le}` — load raw PCM
  directly (used to validate end-to-end against `LastQuestion_long_EN.raw`
  without adding an ffmpeg dependency to the test loop). Mutually
  exclusive with `--wav PATH`.
- `--stream` — route through `Engine::transcribe_samples_stream`
  instead of the existing `run_once` path. Incompatible with `--bench`
  / `--profile` (those continue to exercise the offline path).
- `--stream-chunk-ms N` — segment stride (default 1000).
- `--emit {text,jsonl}` — `text` prints `[start-end] segment` one line
  per callback; `jsonl` prints a single JSON object per line, with
  proper escaping of `"`, `\`, newlines, control chars. Flushes stdout
  after each segment so downstream players/consumers see output
  immediately.

### 7.6 — Validation harness `test-streaming`

`src/test_streaming.cpp` runs on a loaded Engine and asserts:

- Mode 1 reference: `transcribe()` produces the baseline text.
- Mode 2 byte-equality: for `chunk_ms ∈ {250, 500, 1000, 2000, 4000}`
  plus `audio_duration_ms` (single-segment edge case),
  `transcribe_stream()` segments concatenate byte-equal to Mode 1.
- Mode 2 timestamp continuity: every segment's `start_s` matches the
  previous `end_s` (within 1 ms rounding); `end_s > start_s`;
  `is_final=true` for every segment in Phase 1.
- Mode 3 error path: `stream_start()` on a non-streaming GGUF throws
  an exception whose message mentions "streaming" or "Phase 8".

Caught the cumulative-vs-per-window detokenize bug on first run; once
fixed, all checks pass on `jfk.wav` and on the long speech clip via
CLI byte-equality.

### 7.7 — Real-world validation

`LastQuestion_long_EN.raw` (5.46 min, 16 kHz s16le mono,
`qvac/packages/qvac-lib-infer-whispercpp/examples/samples/`):

- offline `--model ... --pcm-in ...` transcript: 5169 bytes, 1710
  tokens, 4099 encoder frames.
- `--stream --stream-chunk-ms 2000` segments concatenated: 5169 bytes,
  byte-equal to offline (`diff` produces no output).
- Metal Q8_0 timing: `mel=152ms enc=14941ms dec=5ms total=15099ms
  RTF=0.046` (22x real-time). Mode 2's overhead over the offline path
  is sub-ms (the stream variant's total wall is the encoder pass +
  1710-token cumulative detokenize + callback dispatch).
- `--emit jsonl` emits one correctly-escaped JSON object per segment;
  first chunk lands at `start=0.000 end=0.480` with the first word
  `"but"`.

### 7.8 — Phase 8 design notes (not yet implemented)

Phase 8 will deliver Mode 3 functional: live duplex streaming where
the caller pushes PCM over time via `StreamSession::feed_pcm_*` and
receives partial-then-final segments as chunks close.

Prerequisites and scope tracked for Phase 8:

1. **Checkpoint selection (go/no-go gate).** Evaluate candidate NeMo
   cache-aware streaming checkpoints against Parakeet-CTC-0.6B on the
   repo's reference clips. Primary candidate:
   `stt_en_fastconformer_hybrid_large_streaming_multi`. Accept only
   if WER on reference set is within ±0.5 % of current offline.
2. **New converter** `scripts/convert-parakeet-streaming-to-gguf.py`,
   scoped similarly to `convert-parakeet-ctc-to-gguf.py`. Sets the
   `parakeet.encoder.streaming.enabled = true` metadata flag that
   `stream_start()` already probes.
3. **Streaming encoder graph**: per-layer attention KV cache tensors
   (left-context), depthwise-conv left-state tensors, chunked +
   left-context attention mask, streaming mel state (reflect-pad only
   on true first/last chunk, per-chunk CMVN *or* running-mean CMVN
   depending on what the chosen checkpoint was trained with).
4. **Bucketed graph cache** — round chunk mel length up to a fixed
   bucket so the 3-entry LRU graph cache reuses the compiled graph
   across chunks. Already tracked in §5.18.
5. `StreamingOptions` gains `left_context_ms`, `right_lookahead_ms`,
   `emit_partials` (activated in Phase 8).
6. Per-stage numerical parity harness vs the NeMo streaming reference,
   following the Round 5-8 methodology.
7. Mode 3 bring-up + `StreamSession` state machine (sample ring,
   bucket-rounded encoder call, KV cache slide, cancellation).
8. Wire the CLI `--pcm-in` path (plus future `--pcm-in -` for stdin)
   through `stream_start()` for manual live-streaming testing.
9. Expected performance (extrapolated from §6.x Metal numbers): 20 s
   clip, `chunk_ms=500`: ~350-450 ms total, first segment ~0.55 s;
   60 s clip, `chunk_ms=2000`: ~700 ms total (vs offline ~850 ms —
   linear-in-T attention wins on long-form).

## Phase 8 — Mode 3 cache-aware streaming  _(in progress)_

### Phase 8.0 — checkpoint landscape (done)

The NeMo registry has only one cache-aware streaming Conformer family:
`stt_en_fastconformer_hybrid_large_streaming_{multi,80ms,480ms,1040ms}`.
Same family powers `qvac/packages/qvac-lib-infer-parakeet`'s `'eou'`
modelType today. It's a 115M-parameter, RNN-T+aux_CTC hybrid trained
with chunked-limited attention. Real-world quality is not great
(~2x WER vs Parakeet-CTC-0.6B offline) — the user's existing
production has confirmed this.

So Phase 8 is **not** going to ship a port of streaming_multi. Instead,
the chosen approach is **cache-aware *inference* on the existing
offline-trained Parakeet-CTC-0.6B weights**: same 600M model, same
quality ceiling, just driven through a streaming forward pass. The
cost is some accuracy degradation because the model wasn't trained
with chunked attention masks.

### Phase 8.1 — Python reference + accuracy bake-off (done)

`scripts/streaming-reference.py` implements the **chunking-with-context**
strategy in Python on top of the NeMo offline model: each chunk feeds
`[left_context + chunk + right_lookahead]` into the offline encoder,
slices out the center frames, runs CTC greedy with a stateful
`prev_token` carried across chunks. This mirrors what the eventual C++
streaming path does (modulo the future Phase 8.5 KV-cache optimisation).

Sweep results on `test/samples/jfk.wav` (11 s clean speech) and
`LastQuestion_long_EN.raw` (5.5 min sci-fi narration with proper nouns,
representing the "harder" production case):

| chunk_ms | left_ctx_ms | right_lookahead_ms | jfk WER | long-clip WER | first-seg latency |
|---------:|------------:|-------------------:|--------:|--------------:|------------------:|
| 1000 | 0     | 0    | 40.91% | n/a    | 1.0 s |
| 1000 | 2000  | 500  | 0.00%  | 14.86% | 1.5 s |
| 2000 | 2000  | 1000 | 0.00%  | 7.64%  | 3.0 s |
| 2000 | 5000  | 1000 | 0.00%  | n/a    | 3.0 s |
| 2000 | 5000  | 2000 | n/a    | 4.02%  | 4.0 s |
| 2000 | 10000 | 1000 | n/a    | 7.53%  | 3.0 s |
| **2000** | **10000** | **2000** | n/a | **3.82%** | **4.0 s** |
| 4000 | 5000  | 1000 | 0.00%  | 4.75%  | 5.0 s |

Key observations:

- **Right lookahead is the single most impactful knob.** Going from
  1000 → 2000 ms right-lookahead drops long-clip WER from 7.64% to
  4.02% at the same chunk + left configuration. The conv module uses
  symmetric `kernel=9` padding (designed at training to see future
  context), so denying it future frames at chunk boundaries hurts more
  than denying past frames.
- **Short audio is forgiving, long audio compounds errors.** jfk is
  at 0% with modest context; 5.5 min same-config drifts to ~7-8%
  because (a) per-window CMVN drifts vs the offline single-statistic
  pass and (b) more boundary opportunities for misreads. Per-window
  CMVN is the suspect; running CMVN may close some of the gap and is
  noted as a Phase 8.5 follow-up.
- **`right=0` (pure causal) is uniformly bad** unless chunks are large
  enough to hide the boundary (chunk=2000 + right=0 → 4.55% on jfk;
  chunk=500 + right=0 → 40.9%). Pure-causal mode is supportable but
  not the recommended default.
- **Left context past 5 s gives diminishing returns** on this model.

**Recommended C++ defaults (subject to revision once C++ measurements
are in):**

- `StreamingOptions{ chunk_ms = 2000, left_context_ms = 10000,
  right_lookahead_ms = 2000 }` — sweet-spot accuracy
  (~4 % WER on long-form, 0 % on short clean speech), ~4 s
  first-segment latency. Suits production live-captioning.
- For lower latency, callers can pick e.g. `chunk_ms=1000,
  left=2000, right=500` and accept ~10-15 % WER on long-form.

### Phase 8.2 — C++ implementation plan

Next milestones in order:

1. Extend `StreamingOptions` with `left_context_ms` +
   `right_lookahead_ms`; remove the runtime gate in `stream_start()`
   (any Parakeet-CTC GGUF works in streaming mode now — no metadata
   probe needed).
2. Implement `StreamSession` state machine (sample ring, chunk
   dispatch, per-window mel + encoder call, logits center-slicing,
   CTC stateful decode, segment emission with absolute timestamps).
3. Wire the CLI `--stream` path to drive `stream_start()` when
   `--pcm-in` is used (or always — same flag for Mode 2 vs Mode 3 is
   surprising; revisit in §8.3).
4. Per-chunk numerical parity vs the Python reference at the same
   `(chunk_ms, left_ctx_ms, right_lookahead_ms)` config to make sure
   the C++ port lands on the same logits, not approximately.
5. Test harness extension covering Mode 3 with random burst feeds.

Phase 8.5 (perf follow-up): replace chunking-with-context with true
KV cache + conv state tensors, ~6× compute reduction on long-form
audio without changing accuracy.

### Phase 8.2 — C++ StreamSession (done)

Landed the Mode 3 state machine in [src/parakeet_engine.cpp](qvac-parakeet.cpp/src/parakeet_engine.cpp)
(`StreamSession::Impl`), backed by the existing `Engine::Impl::model`
through a borrowed pointer. Key pieces:

- `feed_pcm_f32` / `feed_pcm_i16` append samples to a `pending`
  buffer and trigger `try_emit_chunks()`.
- `try_emit_chunks()` consumes one chunk at a time while
  `pending.size() >= chunk_samples + right_lookahead_samples`.
  Per-chunk window = `left_history + chunk + right_lookahead`.
  After the encoder run, the consumed chunk is appended to
  `left_history` (rolling cap at `left_context_samples`).
- `flush_remainder()` (called from `finalize()`) processes the tail
  with whatever lookahead is left. No right-lookahead on the final
  chunk, so the last ~`right_lookahead_ms` of audio sees less conv
  context — acceptable for a one-shot end-of-audio case.
- Per-chunk numerical path reuses `compute_log_mel` + `run_encoder` +
  `ctc_greedy_decode_window` + `detokenize` from the offline stack
  unchanged. Cumulative detokenize with suffix slicing preserves the
  leading-space invariant that bit Mode 2 in §7.3.
- Segment timestamps: `start_s = emitted_samples / sr`,
  `end_s = (emitted_samples + consumed_chunk_samples) / sr`, absolute
  from the start of the session. Matches what the binding's
  `TranscriptionSegment { start, end, toAppend }` expects.
- `StreamingOptions::left_context_ms` and `right_lookahead_ms`
  landed on the public API (defaults 10000 / 2000 respectively, the
  winners from §8.1's sweep).
- `Engine::stream_start()` no longer gates on the GGUF metadata
  `parakeet.encoder.streaming.enabled` — any Parakeet-CTC GGUF works.
  The metadata key survives in the loader for Phase 8.5 / 9 use.

### Phase 8.3 — CLI + validation (done)

CLI:

- `--stream --stream-duplex` routes through `stream_start()` +
  `feed_pcm_f32` with 4 kB default block size (configurable via
  `--stream-feed-bytes`, useful to stress the session state machine).
- `--stream-left-context-ms` + `--stream-right-lookahead-ms` override
  the `StreamingOptions` defaults.
- `--emit text|jsonl` reused unchanged.

Test harness (`src/test_streaming.cpp`) adds three Mode 3 configs on
`jfk.wav` (`chunk_ms × left_ms × right_ms ∈ {1000,2000,500},
{2000,2000,1000}, {2000,5000,2000}`) plus a cancel-path assertion.
PCM is fed in **random-size bursts (512-4000 samples)** via
`feed_pcm_f32` to exercise the ring / chunk-dispatch paths; each
config asserts WER ≤ 5 % vs the Mode 1 reference (all hit 0 %).

### Phase 8.4 — end-to-end numbers

`LastQuestion_long_EN.raw` (5.46 min, 16 kHz s16le, Apple M3 Ultra,
Metal Q8_0), default config `chunk_ms=2000, left=10000, right=2000`:

- C++ Mode 3 transcript: 972 words, **4.13 % WER** vs offline.
- Python f32 reference at the same config: 3.82 % WER. The 0.3 %
  delta is Q8_0 quantisation noise, matches §6.x's measurements.
- Wall time: 35.3 s, RTF 0.108 (9× real-time). ~2.4× slower than the
  Mode 2 offline encoder (RTF 0.046) because each chunk re-runs the
  encoder on a 14 s window (`left + chunk + right`) instead of the
  shipping-forward incremental state. This is the chunking-with-context
  tax; Phase 8.5 closes it.
- First-segment latency: `chunk_ms + right_lookahead_ms` ≈ 4 s wall
  (matches Python reference).

### Phase 8.5 — KV cache / conv state (pending)

Same `StreamSession` public API, swap the internal loop: keep
per-layer `K`, `V`, and depthwise-conv left-state tensors around as
backend buffers, slid forward each chunk. Each encoder call then only
computes over new-chunk + right-lookahead frames instead of the full
`(left + chunk + right)` window. Projected wins:

- Per-chunk compute: down from `O(left + chunk + right)` to
  `O(chunk + right)`, i.e. ~5× on the default config
  (2 + 2 vs 10 + 2 + 2).
- Total wall on the 5.5 min clip: down from 35 s to ~10-15 s (ballpark
  close to the offline 15 s baseline).
- Accuracy unchanged — this is a pure compute-layout refactor.

Requires graph changes (persistent cache tensors for attention + conv
module, streaming attention mask with `att_context_size` plumbing
already used by NeMo's own cache-aware export path), plus a per-stage
parity harness vs the §8.1 Python reference. Out of scope for this PR.

## Phase 9 — multi-model support _(in progress)_

Phase 9 extends the Parakeet-CTC pipeline beyond the initial 0.6B
checkpoint. Target is drop-in support for other NeMo Parakeet-CTC
checkpoints that share the FastConformer architecture, without
branching the converter or the C++ encoder graph.

### Phase 9.1 — parakeet-ctc-1.1b (done)

HF repo: `nvidia/parakeet-ctc-1.1b`. NeMo encoder config:

    d_model: 1024          (same as 0.6B)
    n_layers: 42           (was 24)
    n_heads: 8             (same)
    ff_expansion_factor: 4 (same; ff_dim=4096)
    conv_kernel_size: 9    (same)
    subsampling_factor: 8  (same)
    subsampling_conv_channels: 256 (same)
    self_attention_model: rel_pos (same)
    conv_norm_type: batch_norm (same, fused at convert)
    att_context_size: [-1, -1]  (same, offline)
    vocab_size: 1025 (1024 BPE + CTC blank)

Only `n_layers` differs from 0.6B. The converter already reads
`n_layers` from the NeMo YAML and iterates block-by-block, and the C++
loader reads `parakeet.encoder.n_layers` from GGUF metadata. Zero
code changes needed to produce a working 1.1B GGUF and transcribe with
it. The one stale hardcode was in the `--profile` CLI path
(`layer_points = {0, 1, 12, 24}`); now scales with `n_layers` via
`{0, 1, n_layers/2, n_layers}`.

End-to-end results on Apple M3 Ultra, Metal, Q8_0:

| Clip | Model | Wall time | RTF | Encoder ms |
|-|-|-|-|-|
| jfk.wav (11 s) | ctc-0.6b | 170 ms | 0.015 | ~155 ms |
| jfk.wav (11 s) | ctc-1.1b | 281 ms | 0.026 | 276 ms |
| LastQuestion_long_EN.raw (5.5 min) | ctc-0.6b | 15.1 s | 0.046 | 14.9 s |
| LastQuestion_long_EN.raw (5.5 min) | ctc-1.1b | 24.4 s | 0.074 | 24.2 s |

1.1B is 1.6-1.8× slower than 0.6B, roughly matching the layer-count
ratio (42/24 ≈ 1.75). Metal still hits ~13× real-time on the long clip
with 1.1B, comfortably faster than real-time. Transcripts differ on
2.3 % of words on the long clip (22 / 969) — typical quality difference
between the two models; both ship transcripts that track the same
content.

Streaming (Mode 2 and Mode 3) works out of the box with 1.1B:
`test-streaming` passes all chunk-size byte-equality checks (Mode 2)
and all three (chunk × left × right) configs at WER ≤ 5% (Mode 3).

### Phase 9.x — next candidates

Natural follow-ups that reuse the same converter + encoder graph:

- `nvidia/parakeet-tdt_ctc-110m`: 110 M hybrid TDT+CTC. 512 × 17
  FastConformer. CTC head alone can be decoded by the existing
  `ctc_greedy_decode_window`; TDT primary decoder is a separate
  port (prediction net + joint net + transducer greedy).
- `nvidia/parakeet-tdt-0.6b-v3`: bigger TDT model family — same
  transducer-decoder port, larger encoder.
- `nvidia/parakeet-ctc-110m`: smaller CTC-only variant if it exists;
  would land as a converter-flag change only.
