# qvac-parakeet.cpp

**Parakeet-CTC** (NVIDIA, CC-BY-4.0 FastConformer ASR family) ported to
[`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference on CPU
and GPU (Metal / CUDA / Vulkan), with no runtime dependency on Python,
PyTorch, or onnxruntime. TDT / EOU / Sortformer pipelines land as
follow-ups.

Supported checkpoints:

| HF repo | Decoder | Mel | `d_model × n_layers` | Vocab | Params | GGUF size | RTF (Metal) | Languages |
|-|-|-|-|-|-|-|-|-|
| `nvidia/parakeet-ctc-0.6b`    | CTC  | 80  | 1024 × 24 | 1024 | 600 M  | 697 MiB q8_0 / 1.3 GiB f16  | 0.014-0.046 | English only |
| `nvidia/parakeet-ctc-1.1b`    | CTC  | 80  | 1024 × 42 | 1024 | 1.1 B  | 1217 MiB q8_0               | 0.026-0.074 | English only |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT  | 128 | 1024 × 24 | 8192 | 600 M  | 715 MiB q8_0 / 1.34 GiB f16 | 0.024-0.050 | ~25 languages + PnC |

Same converter, same encoder graph (biases go through an optional
path when the checkpoint sets `use_bias=False`), same GGUF schema.
Model identity lives entirely in `parakeet.model.type` + the encoder
hyperparameters.

The TDT decoder (prediction net + joint net + transducer greedy) runs
on CPU in pure float32 after dequantizing its ~70 MiB of weights once
at Engine construction. One-shot transcription only for now
(`Engine::transcribe()` / `--wav` / `--pcm-in`); Mode 2/3 streaming is
still CTC-only.

Mirrors [`chatterbox.cpp`](https://github.com/GustavoA1604/chatterbox.cpp)'s
layout and staged-validation methodology, so contributors familiar with
that repo will find the same file structure here.

---

## Pipeline at a glance

```
      16 kHz mono wav                                       text
             |                                                ^
             v                                                |
  +------------------------------------------------------------+
  |                        qvac-parakeet                        |
  |                                                             |
  |    wav  ->  80-ch log-mel  ->  FastConformer encoder        |
  |             (STFT + CMVN)      (subsampling 8x + 24 blocks) |
  |                                                             |
  |                 ->  CTC head  ->  greedy decode  ->  text   |
  +------------------------------------------------------------+
             ^                                                |
             |                                                v
       dr_wav reader                              SentencePiece BPE
                                                  (embedded in GGUF)
```

Everything is self-contained in one `.gguf` file: encoder weights,
CTC head, precomputed mel filterbank, and the SentencePiece tokenizer.

## Prerequisites

- C++17 compiler (clang or gcc)
- cmake >= 3.14
- Python 3.10+ with `torch`, `nemo_toolkit[asr]`, `gguf`, `numpy`,
  `librosa`, `soundfile`, `sentencepiece` — needed **once**, at setup
  time only, to run the weight converter (which bakes the precomputed
  mel filterbank into the GGUF) and the reference-dump scripts. Once
  the GGUF exists, the C++ binary has zero runtime dependency on Python.

See `scripts/` for one-shot helpers.

## 1. Clone and build

```bash
git clone <this-repo> qvac-parakeet.cpp
cd qvac-parakeet.cpp

# Clone ggml at the pinned commit (CPU-only; no GPU patches in phase 1).
./scripts/setup-ggml.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

For a GPU backend — Metal on Apple Silicon (**~2.5x faster than CPU**),
CUDA on NVIDIA, Vulkan elsewhere:

```bash
# Apple Silicon:
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
# or: -DGGML_CUDA=ON / -DGGML_VULKAN=ON
cmake --build build-metal -j$(sysctl -n hw.ncpu)

# Run — pass any value > 0 to --n-gpu-layers to enable GPU
# (the whole encoder runs on one backend; the value is currently a
# yes/no toggle, named for compat with llama.cpp / whisper.cpp convention):
./build-metal/qvac-parakeet \
    --n-gpu-layers 1 \
    --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --wav   test/samples/jfk.wav
```

This produces the main binary plus per-stage validation harnesses:

| Binary                  | What it does |
|-------------------------|--------------|
| `build/qvac-parakeet`   | End-to-end: wav / raw PCM -> text (FastConformer + CTC greedy decode + SentencePiece detokenize), with optional `--stream` Mode 2 output. |
| `build/test-mel`        | 16 kHz 80-ch log-mel parity vs NeMo `AudioToMelSpectrogramPreprocessor`. |
| `build/test-encoder`    | FastConformer encoder per-stage parity vs `dump-ctc-reference.py`. |
| `build/test-ctc`        | CTC head + greedy decode parity vs NeMo `transcribe()`. |
| `build/test-streaming`  | Mode 2 byte-equality + timestamp coverage + Mode 3 error-path assertions across chunk sizes {250, 500, 1000, 2000, 4000} ms. |

## 2. One-time: convert weights

```bash
python -m venv venv && . venv/bin/activate
pip install "nemo_toolkit[asr]" gguf numpy soundfile librosa sentencepiece

python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt models/parakeet-ctc-0.6b.nemo \
  --out  models/parakeet-ctc-0.6b.gguf

# or the bigger 1.1B variant (same converter, same flags)
python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt models/parakeet-ctc-1.1b.nemo \
  --out  models/parakeet-ctc-1.1b.q8_0.gguf \
  --quant q8_0
```

The script downloads `nvidia/parakeet-ctc-0.6b` (or `-1.1b`) from
Hugging Face on first run if the local path doesn't exist. The
SentencePiece tokenizer (`tokenizer.model`) and the precomputed mel
filterbank are embedded directly into the GGUF as standard
`tokenizer.ggml.*` metadata and a named `preproc/mel_filterbank`
tensor, so the C++ binary is self-contained.

### Quantization tiers

`--quant` selects the storage format for the ~150 large 2D weight
matrices (FFN, attention q/k/v/out/pos/qkv, conv pointwise, subsampling
output, CTC head). Small tensors (biases, norms, fused BN, mel
filterbank, depthwise/ small 2D convs) always stay at f32/f16.

| `--quant` | File size | enc best on 20 s clip | enc best on 11 s clip | Transcript parity |
|-----------|-----------|----------------------:|----------------------:|-------------------|
| `f32`     | 2.4 GiB   | n/a (debug only)      | n/a                   | exact            |
| `f16`     | 1.3 GiB   | 1221 ms               | ~680 ms               | bit-equal        |
| `q8_0`    | 697 MiB   | **839 ms**            | **460 ms**            | bit-equal        |
| `q5_0`    | 453 MiB   | 1475 ms (slower)      | ~650 ms               | bit-equal        |
| `q4_0`    | 372 MiB   | 1080 ms               | 595 ms                | bit-equal        |

Measurements on an Apple M3 Ultra, 10 ggml-cpu threads, OpenMP,
`--bench-warmup 5 --bench-runs 15`. Transcripts on both clips are
bit-equal to NeMo PyTorch reference at every tier tested, including
`q4_0`.

**Recommended defaults** on current CPU hardware:
- `q8_0` — the speed + accuracy sweet spot. 11 % faster than
  onnxruntime on 20 s audio best-case encoder, 23 % faster on 11 s.
  Model 2x smaller than f16.
- `q4_0` — smallest runnable variant (3.5x smaller than f16), still
  bit-equal transcripts on clean speech.

`q5_0` ships as well but the ggml-cpu `q5_0` mul_mat kernel is slower
than either `q8_0` or `q4_0` on Apple Silicon, so it's only useful if
you want the `q5_0` size tier specifically.

### Reference comparison vs onnxruntime (20 s clip, sample-16k.wav, 5 warmup + 15 timed runs)

**CPU f16 vs f16** — same floating-point precision, different runtimes:

```
                   onnxruntime-f16    ggml-cpu-f16
  -----------------------------------------------
  model size           2.3 GiB         1.3 GiB
  load ms              16 736            642      (26x faster)
  inf best ms             948           1117      (15 % slower)
  inf median ms         1 007           1132      (12 % slower)
  inf stdev ms             52             18      (3x tighter)
  RTF best               0.047          0.055
  RTF median             0.050          0.056
  Transcripts            match          match
```

**CPU int8 vs int8** — same quantization level, different runtimes:

```
                   onnxruntime-int8    ggml-cpu-q8_0
  -------------------------------------------------
  model size          583.9 MiB         697 MiB
  load ms               2 054             179      (11x faster)
  inf best ms             677             898      (25 % slower)
  inf median ms           721             928      (22 % slower)
  inf stdev ms             55              25      (2x tighter)
  RTF best               0.034           0.045
  RTF median             0.036           0.046
  Transcripts            match           match
```

**GPU Metal** — same GGUF, Metal backend (`-DGGML_METAL=ON`, `PARAKEET_BACKEND=metal`):

```
                   onnxruntime-int8    ggml-metal-q8_0
  ---------------------------------------------------
  model size          583.9 MiB         697 MiB
  load ms               2 295              420      (5.5x faster)
  inf best ms             682              282      (2.4x faster)
  inf median ms           712              283      (2.5x faster)
  inf stdev ms             18             0.83      (21x tighter)
  RTF best               0.034           0.014
  RTF median             0.035           0.014
  Transcripts            match           match      (73x real-time!)
```

On CPU, onnxruntime uses AMX-accelerated kernels and is 12–25 %
faster on raw throughput. On Metal (Apple Silicon GPU), ggml is
**2.4–2.5× faster** than onnxruntime int8 with 21× tighter run-to-run
variance (0.83 ms stdev vs 18 ms). Metal inference is compute-bound
on shader units, so the choice of quant tier (f16 / Q8_0 / Q4_0) only
affects file size — all three land at ~272 ms encoder on a 20 s clip.

## 3. Run - wav -> text

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --wav   test/samples/jfk.wav
```

### Raw PCM input

For headless pipelines (ffmpeg / sox upstream, or the QVAC bindings), the
CLI also accepts raw 16 kHz mono PCM via `--pcm-in`:

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --pcm-in recording.raw \
    --pcm-format s16le        # or f32le; defaults to s16le
```

### Streaming — Mode 2 (full audio in, segments streamed out)

The engine exposes three transcription entry points that mirror the qvac
SDK's `transcribe` / `transcribeStream` API:

| Entry point | Caller provides | Caller receives | Status |
|-|-|-|-|
| `Engine::transcribe()` | full audio | full text | ships |
| `Engine::transcribe_stream()` | full audio + callback | segments via callback | **ships (Mode 2)** |
| `Engine::stream_start()` -> `StreamSession` | push PCM via `feed_pcm_*()` | segments via callback | **ships (Mode 3, cache-aware inference)** |

Mode 2 runs the offline encoder once, then walks CTC frames in
`chunk_ms`-sized windows and emits one `StreamingSegment` per window via
the callback. Transcript is **byte-equal** to the non-streaming path —
`test-streaming` asserts this across chunk sizes {250, 500, 1000, 2000,
4000} ms on every run.

From the CLI:

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --pcm-in recording.raw --pcm-format s16le \
    --stream --stream-chunk-ms 1000 \
    --emit text         # or jsonl
```

Flags:

- `--stream` — enable Mode 2.
- `--stream-chunk-ms N` — segment window stride (default 1000; snaps
  down to multiples of the 80 ms encoder frame stride).
- `--emit text` — one `[start-end] text` line per segment (default).
- `--emit jsonl` — one `{"chunk","start","end","is_final","text"}` JSON
  object per line, for easy downstream consumption.

Observed on an Apple M3 Ultra (Metal Q8_0) feeding a 5.5 minute speech
clip (`LastQuestion_long_EN.raw`, 16 kHz s16le):

```
audio=327.91s samples=5246635@16000Hz mel_frames=32792 enc_frames=4099
mel=152ms enc=14941ms dec=5ms total=15099ms RTF=0.046 tokens=1710
```

Segments are emitted to stdout at the `--stream-chunk-ms` cadence once
the offline encoder finishes. Mode 2 is *cosmetic streaming*: first
segment lands after the full encoder pass.

### Streaming — Mode 3 (live duplex, cache-aware inference)

Mode 3 feeds PCM into a `StreamSession` incrementally; each chunk runs
its own encoder pass over `[left_context + chunk + right_lookahead]`
audio, slices out the center frames, and emits a segment as soon as
that chunk is processed. First segment lands at
`chunk_ms + right_lookahead_ms + encoder_time`, not after the full
utterance.

Key point: **no new model needed**. Mode 3 runs the existing
offline-trained 600M Parakeet-CTC-0.6B weights in cache-aware inference
mode. Accuracy is preserved within a few percent of offline WER when
the `left_context_ms` and `right_lookahead_ms` budgets are reasonable
(the conv module uses symmetric `kernel=9` padding, so denying future
context at chunk boundaries hurts more than denying past context).

From the CLI (simulates a live producer feeding the same wav in blocks):

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --pcm-in recording.raw --pcm-format s16le \
    --stream --stream-duplex \
    --stream-chunk-ms          2000 \
    --stream-left-context-ms   10000 \
    --stream-right-lookahead-ms 2000 \
    --emit text         # or jsonl
```

Mode 3 knobs (all in `StreamingOptions` on the C++ side,
`--stream-*-ms` on the CLI):

- `chunk_ms` — audio stride at which segments are emitted.
- `left_context_ms` — past audio prepended to the encoder input each
  chunk. 10 s is a solid default; diminishing returns past 5 s.
- `right_lookahead_ms` — future audio appended before emitting the
  chunk; most impactful accuracy knob.

Measured on Apple M3 Ultra, Q8_0, Metal backend:

| Audio | Config (chunk / left / right ms) | WER vs offline | Wall time | First-seg latency |
|-|-|-|-|-|
| `jfk.wav` (11 s) | 1000 / 2000 / 500 | **0.00%** | ~1.8 s | ~1.6 s |
| `jfk.wav` (11 s) | 2000 / 2000 / 1000 | **0.00%** | ~1.8 s | ~3.1 s |
| `jfk.wav` (11 s) | 2000 / 5000 / 2000 | **0.00%** | ~1.9 s | ~4.1 s |
| `LastQuestion_EN.raw` (5.5 min) | 2000 / 10000 / 2000 | **4.13%** | 35 s (RTF 0.11) | ~4 s |

Mode 3 is slower in total wall time than Mode 2 because each chunk
re-runs the encoder over the full `(left_ctx + chunk + right_lookahead)`
window. A KV-cache + conv-state optimisation (Phase 8.5) will roughly
6x the per-chunk compute on long-form audio while preserving the same
accuracy; the `StreamSession` public API already supports it as a
drop-in swap.

The Node binding at
[qvac-lib-infer-parakeet](https://github.com/qvac/qvac-lib-infer-parakeet)
drives `StreamSession` directly from its existing `append({type:'audio',
data})` flow — each incoming `Buffer` maps to `feed_pcm_i16`, and
`{type:'end of job'}` maps to `finalize()`.

### Live microphone example

`examples/live-mic.cpp` wraps `StreamSession` around
[`miniaudio`](https://miniaud.io/) (single-header, MIT, vendored under
`examples/miniaudio.h`) for real-time transcription from the default
capture device on macOS / Linux / Windows. Terminal output only, no GUI.

```bash
# Built as part of the default CLI target set; gated on
# -DQVAC_PARAKEET_BUILD_EXAMPLES=ON (on by default when the project
# is the top-level CMake).

# List capture devices:
./build-metal/live-mic --list-devices

# Transcribe live (Ctrl-C to stop, Metal backend recommended):
./build-metal/live-mic \
    --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --n-gpu-layers 1 \
    --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000
```

First time you run it macOS will prompt for microphone access. The
capture thread pushes f32 samples into a mutex-guarded queue; the main
thread drains the queue and calls `StreamSession::feed_pcm_f32`, so
the encoder runs off the audio callback thread (no capture-buffer
stalls). Ctrl-C sets a stop flag, the capture device is stopped, the
tail buffer is flushed, `finalize()` emits the last segment, and the
binary exits cleanly.

Defaults chosen for an interactive feel: first segment lands ~2 s
after you start speaking (`chunk_ms + right_lookahead_ms`), segments
afterward at the `chunk_ms` cadence.

## 4. Optional: validate against NeMo PyTorch

```bash
# One-time: dump NeMo reference tensors from the same wav.
python scripts/dump-ctc-reference.py \
    --wav test/samples/jfk.wav \
    --out artifacts/ctc-ref

# C++ parity harnesses.
./build/test-mel     test/samples/jfk.wav artifacts/ctc-ref/mel.npy
./build/test-encoder models/parakeet-ctc-0.6b.gguf artifacts/ctc-ref
./build/test-ctc     models/parakeet-ctc-0.6b.gguf artifacts/ctc-ref/logits.npy
```

Expected per-stage rel error (NeMo PyTorch vs C++ at `--quant f16`):

```
Stage A  log_mel               ~ 1e-4 inner / ~ 2e-3 boundary (f32 FFT)
Stage B  subsampling_out       rel ~ 1e-3 (f16 quantization floor)
Stage C  block_0_out           rel ~ 1e-3
Stage D  block_23_out          rel ~ 2e-3
Stage E  ctc_logits            rel ~ 1e-3
Stage F  decoded transcript    edit distance = 0 on clean speech
```

At `--quant q8_0` through `q4_0` the per-stage rel inflates by ~3x
to ~25x, but the transcript stays bit-equal on clean speech. See
`PROGRESS.md` 5.12 for the sweep results.

## Current status

Phases 0 through 7 are complete:

- `qvac-parakeet --model ... --wav ...` produces the expected
  transcript end-to-end, matching NeMo PyTorch bit-equivalently on
  `jfk.wav` and `sample-16k.wav` at every quant tier (f16 through
  Q4_0) on both CPU and Metal backends.
- Per-stage numerical parity is at the f16 quantization floor
  (1–2e-3 rel vs NeMo PyTorch) on every intermediate encoder tensor.
- **CPU Q8_0**: encoder runs 22x real-time on an M3 Ultra CPU.
  Faster than ONNX f16 by 12 %, slower than ONNX int8 by 22 %.
- **Metal Q8_0**: encoder runs **73x real-time** on the M3 Ultra
  GPU. **2.5x faster than onnxruntime int8** with 21x tighter
  variance (0.83 ms stdev).
- **Phase 7 — Mode 2 streaming output**: `Engine::transcribe_stream()`
  walks CTC frames in `chunk_ms` windows and emits per-segment
  callbacks, byte-equal to the offline transcript. `--stream` CLI
  flag + `--pcm-in` raw input + `--emit text|jsonl`.
- **Phase 8 — Mode 3 live duplex streaming (cache-aware inference)**:
  `Engine::stream_start()` -> `StreamSession` with `feed_pcm_f32/i16` +
  `finalize()`. Uses the **existing offline 600M GGUF** in a
  chunking-with-context streaming pass, so no new model is needed. On a
  5.5 min sci-fi clip, Mode 3 transcribes at ~4 % WER vs offline with
  ~4 s first-segment latency at default settings. `--stream-duplex`
  CLI + `--stream-left-context-ms` + `--stream-right-lookahead-ms`.
- See PROGRESS.md for the round-by-round journal (§5.11–5.17 for
  Rounds 5–8, §6.x for the Metal bring-up, §7.x for Mode 2 streaming,
  §8.x for Mode 3 cache-aware streaming).

- **Phase 10 — TDT (Token-and-Duration Transducer)**: multilingual
  port of `nvidia/parakeet-tdt-0.6b-v3` — 2-layer LSTM prediction
  net + joint MLP + transducer greedy decode running on CPU in f32
  after dequantization. Byte-identical to NeMo on jfk.wav with
  proper capitalization + punctuation; clean multilingual output on
  es/fr/de/it/pt/ru samples. One-shot transcription only; TDT in
  streaming modes tracked in PROGRESS.md §10.5.

Next: Phase 8.5 (true KV cache + conv state for ~6x compute reduction on
long-form audio without accuracy change), TDT in streaming (Phase 10.5).
Then `CONV_2D_DW` on Metal (upstream ggml contribution), Metal
flash-attn, EOU / Sortformer pipelines.

## Repository layout

```
qvac-parakeet.cpp/
  ggml/                          pristine ggml clone (not tracked; populated
                                   by scripts/setup-ggml.sh, or skipped entirely
                                   when building with -DQVAC_PARAKEET_USE_SYSTEM_GGML=ON)
  src/
    main.cpp                     CLI (wav / raw PCM -> text, + streaming) + qvac_parakeet_cli_main impl
    cli_main.cpp                 thin main() -> qvac_parakeet_cli_main shim
    parakeet_ctc.{h,cpp}         FastConformer encoder + CTC head ggml graph + GGUF loader
    parakeet_engine.cpp          Engine + StreamSession implementation (transcribe, transcribe_stream, stream_start)
    mel_preprocess.{h,cpp}       wav I/O + STFT + mel + CMVN
    sentencepiece_bpe.{h,cpp}    SentencePiece BPE detokenizer
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare
    test_*.cpp                   per-stage numerical-parity harnesses + streaming validation
  include/qvac-parakeet/
    qvac-parakeet.h              CLI entry (qvac_parakeet_cli_main)
    ctc/pipeline.h               one-shot wav -> text API
    ctc/engine.h                 persistent Engine (load once, transcribe many)
  scripts/
    setup-ggml.sh                pin + clone ggml
    convert-parakeet-ctc-to-gguf.py   .nemo -> GGUF
    dump-ctc-reference.py        NeMo PyTorch -> .npy reference tensors
    transcribe.sh                wav -> text wrapper
  cmake/                         CMake package config (for vcpkg follow-up)
  PROGRESS.md                    chronological development journal
  README.md                      this file
```

## License

Released under the [Apache License 2.0](LICENSE).

**Model license**: the Parakeet-CTC-0.6B weights are licensed
[CC-BY-4.0 by NVIDIA](https://huggingface.co/nvidia/parakeet-ctc-0.6b).
This repository only ships the inference code; model weights are
downloaded on demand.

The bundled `ggml/` is MIT-licensed (see `ggml/LICENSE`).
