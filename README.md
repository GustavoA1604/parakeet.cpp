# qvac-parakeet.cpp

**Parakeet** (NVIDIA, CC-BY-4.0 FastConformer ASR family) ported to
[`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference on CPU
and GPU (Metal / CUDA / Vulkan), with no runtime dependency on Python,
PyTorch, or onnxruntime. Ships CTC, TDT, and Sortformer engines today
under one `Engine` umbrella; EOU pipelines are the next workstream.

Supported checkpoints:

| HF repo | Decoder | Mel | `d_model × n_layers` | Vocab | Params | GGUF size | RTF (Metal) | Languages |
|-|-|-|-|-|-|-|-|-|
| `nvidia/parakeet-ctc-0.6b`    | CTC  | 80  | 1024 × 24 | 1024 | 600 M  | 697 MiB q8_0 / 1.3 GiB f16  | 0.014-0.046 | English only |
| `nvidia/parakeet-ctc-1.1b`    | CTC  | 80  | 1024 × 42 | 1024 | 1.1 B  | 1217 MiB q8_0               | 0.026-0.074 | English only |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT  | 128 | 1024 × 24 | 8192 | 600 M  | 715 MiB q8_0 / 1.34 GiB f16 | 0.024-0.050 | ~25 languages + PnC |
| `nvidia/parakeet-tdt-1.1b`    | TDT  | 80  | 1024 × 42 | 1024 | 1.1 B  | 1225 MiB q8_0               | 0.027-0.079 | English only, lowest WER (no PnC) |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer head (diarization) | 80 | enc 512 × 18 + tf 192 × 18 | n/a (4 speakers) | ~123 M | 263 MiB f16 | 0.017-0.097 | Speaker diarization (up to 4 speakers, offline) |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer head (diarization) | 128 | enc 512 × 17 + tf 192 × 18 | n/a (4 speakers) | ~117 M | 251 MiB f16 | similar to v1 in offline mode | Speaker diarization, streaming-trained (offline + Phase 11.11.1 sliding-history live streaming today; full NeMo-style spkcache streaming in Phase 11.11.2) |

Same converter, same encoder graph (biases go through an optional
path when the checkpoint sets `use_bias=False`), same GGUF schema.
Model identity lives entirely in `parakeet.model.type` + the encoder
hyperparameters.

All five public entry points sit on a single `qvac_parakeet::Engine`
that auto-dispatches on `parakeet.model.type`:

- `Engine::transcribe()` -- one-shot wav -> text. CTC or TDT.
- `Engine::transcribe_stream()` -- Mode 2, offline encoder + streamed
  segments. CTC or TDT.
- `Engine::stream_start()` -> `StreamSession` -- Mode 3, live duplex
  cache-aware push API. CTC or TDT. TDT needs slightly more context
  at the same chunk size (transducer is more sensitive to missing
  right-lookahead; typical WER delta vs offline is +5-10 %).
- `Engine::diarize()` -- one-shot wav -> [{speaker, start, end}].
  Sortformer.
- `Engine::diarize_start()` -> `SortformerStreamSession` -- live
  diarization push API (sliding-history v1; Phase 11.11.2 spkcache
  pending). Sortformer.

Plus a free function `transcribe_with_speakers(sortformer_engine,
asr_engine, ...)` for combined "who said what" attribution.

---

## Pipeline at a glance

```
  wav -> log-mel (80/128) -> FastConformer encoder (sub 8x, 17-42 blocks)
                                    |
        +---------------------------+---------------------------+
        v                           v                           v
   CTC head + greedy           TDT LSTM + joint MLP +     Sortformer encoder_proj +
        + SP detok               transducer greedy        18L TF + sigmoid head
        |                           |                           |
       text                  text + PnC                {speaker, t0, t1}*
```

Each `.gguf` ships everything its decoder needs in a single file
(encoder weights, decoder weights, precomputed mel filterbank, and
the SentencePiece tokenizer where applicable). The C++ `Engine`
auto-detects the model type at load time and dispatches to the right
decoder, so the public API stays single-engine from the consumer's
perspective.

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

# Clone ggml at the pinned commit. The same pin is used for every
# backend (CPU, Metal, CUDA, Vulkan); no engine- or backend-specific
# ggml patches are applied today.
./scripts/setup-ggml.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)
```

For a GPU backend pick **one** of Metal (Apple Silicon, **~2.5x faster
than CPU**), CUDA (NVIDIA), or Vulkan (everything else) at configure
time. The init order at runtime is `CUDA -> Metal -> Vulkan -> CPU`,
so a single binary built with multiple backends compiled in will use
the first available one and there is no runtime backend switch -- the
expectation is one backend per build.

```bash
# Apple Silicon:
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
# NVIDIA:   -DGGML_CUDA=ON
# Generic:  -DGGML_VULKAN=ON
cmake --build build-metal -j$(sysctl -n hw.ncpu)

# `--n-gpu-layers` is a yes/no toggle today: any value > 0 moves the
# whole encoder to the compiled-in GPU backend. The flag is named for
# compatibility with llama.cpp / whisper.cpp; partial-layer offload
# is not implemented (encoder is small enough to fit on one device).
./build-metal/qvac-parakeet \
    --n-gpu-layers 1 \
    --model models/parakeet-ctc-0.6b.gguf \
    --wav   test/samples/jfk.wav
```

(Use a quantised GGUF -- e.g. `parakeet-ctc-0.6b.q8_0.gguf` --
produced via `--quant q8_0` in the converter snippet under §2 if you
want the smaller / faster file. The bare `.gguf` is f16 and works
the same.)

This produces the main binary plus per-stage validation harnesses:

| Binary                            | What it does |
|-----------------------------------|--------------|
| `build/qvac-parakeet`             | End-to-end CLI: wav / raw PCM -> text (CTC + TDT) or speaker segments (Sortformer). Auto-routes on GGUF metadata. Supports `--stream` (Mode 2/3 transcription, sliding-history Sortformer streaming), `--diarization-model PATH` (combined ASR + Sortformer attribution), `--bench`, `--profile`. |
| `build/live-mic`                  | Live microphone session for either transcription (CTC/TDT) or diarization (Sortformer). Auto-detects from the GGUF. |
| `build/live-mic-attributed`       | Live microphone with simultaneous ASR + Sortformer; tags each transcript segment with the speaker whose live diarization range overlaps it the most. `--accumulate` collapses output to one line per speaker. |
| `build/test-mel`                  | 16 kHz log-mel parity vs NeMo `AudioToMelSpectrogramPreprocessor`. |
| `build/test-encoder`              | FastConformer encoder per-stage parity vs `dump-ctc-reference.py`. |
| `build/test-ctc`                  | CTC head + greedy decode + SentencePiece detokenize parity vs NeMo `transcribe()` (consumes `logits.npy` from `dump-ctc-reference.py`). |
| `build/test-tdt-encoder-parity`   | TDT encoder per-stage parity vs `dump-tdt-reference.py`. |
| `build/test-sortformer-parity`    | Sortformer mel + encoder + speaker-prob parity vs `dump-sortformer-reference.py`. |
| `build/test-streaming`            | CTC/TDT Mode 2 byte-equality + timestamp coverage + Mode 3 WER tolerance across chunk sizes. |
| `build/test-sortformer-streaming` | `SortformerStreamSession` push API: random-burst feed, no-duplicate, single-`is_final` assertions. |

### Build options worth knowing

- `-DQVAC_PARAKEET_BUILD_TESTS=ON` (default ON in standalone): builds
  the `test-*` parity + streaming harnesses listed above.
- `-DQVAC_PARAKEET_BUILD_EXAMPLES=ON` (default `QVAC_PARAKEET_STANDALONE_DEFAULT`,
  i.e. ON for top-level `cmake -S . -B build` but OFF when consumed as
  a sub-project): builds `live-mic` + `live-mic-attributed`.
- `-DQVAC_PARAKEET_USE_SYSTEM_GGML=ON`: link against an installed
  ggml instead of the pinned clone in `ggml/`.
- `-DGGML_METAL=ON` / `-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`: pick
  exactly one GPU backend at configure time (see GPU note above).

## 2. One-time: convert weights

The converter (`scripts/convert-parakeet-ctc-to-gguf.py` -- name is
historical; it auto-detects CTC, TDT and Sortformer from the .nemo
config and writes the right GGUF in each case) takes a `.nemo` archive
and produces a single self-contained GGUF (encoder + decoder weights +
embedded tokenizer where applicable + precomputed mel filterbank).

```bash
python -m venv venv && . venv/bin/activate
pip install "nemo_toolkit[asr]" gguf numpy soundfile librosa sentencepiece

# Parakeet-CTC 0.6B / 1.1B (English, fast)
python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt models/parakeet-ctc-0.6b.nemo \
  --out  models/parakeet-ctc-0.6b.gguf

python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt models/parakeet-ctc-1.1b.nemo \
  --out  models/parakeet-ctc-1.1b.q8_0.gguf \
  --quant q8_0

# Parakeet-TDT 0.6B-v3 / 1.1B (multilingual, punctuation, capitalisation)
python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt    models/parakeet-tdt-0.6b-v3.nemo \
  --hf-repo nvidia/parakeet-tdt-0.6b-v3 \
  --out     models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --quant   q8_0

python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt    models/parakeet-tdt-1.1b.nemo \
  --hf-repo nvidia/parakeet-tdt-1.1b \
  --out     models/parakeet-tdt-1.1b.q8_0.gguf \
  --quant   q8_0

# Sortformer 4-speaker diarization (offline v1, streaming-trained v2)
python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt    models/diar_sortformer_4spk-v1.nemo \
  --hf-repo nvidia/diar_sortformer_4spk-v1 \
  --out     models/sortformer-4spk-v1.f16.gguf

python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt    models/diar_streaming_sortformer_4spk-v2.nemo \
  --hf-repo nvidia/diar_streaming_sortformer_4spk-v2 \
  --out     models/sortformer-streaming-4spk-v2.f16.gguf
```

Footgun: the script's `--hf-repo` defaults to `nvidia/parakeet-ctc-0.6b`,
so when `--ckpt` points at a non-CTC path that does not exist locally
**you must pass `--hf-repo` explicitly** -- otherwise the script will
download the CTC checkpoint instead of the one named in `--ckpt`.

`scripts/download-all-models.sh` pre-fetches every supported `.nemo`
(plus the corresponding ONNX bundles for the Node binding) -- handy
when you're about to be on a flaky network.

### Quantization tiers

`--quant` selects the storage format for the ~150 large 2D weight
matrices (FFN, attention q/k/v/out/pos/qkv, conv pointwise, subsampling
output, CTC head). Small tensors (biases, norms, fused BN, mel
filterbank, depthwise/ small 2D convs) always stay at f32/f16. Tensors
whose `shape[-1] % 32 != 0` also stay at f16 (the block-quantised
formats need a multiple-of-32 last dim); see `PROGRESS.md §5.12` for
the full sweep + tier-by-tier accuracy.

| `--quant` | File size | enc best on 20 s clip | enc best on 11 s clip | Transcript parity |
|-----------|-----------|----------------------:|----------------------:|-------------------|
| `f32`     | 2.4 GiB   | n/a (debug only)      | n/a                   | exact            |
| `f16`     | 1.3 GiB   | 1221 ms               | ~680 ms               | bit-equal        |
| `q8_0`    | 697 MiB   | **839 ms**            | **460 ms**            | bit-equal        |
| `q5_0`    | 453 MiB   | 1475 ms (slower)      | ~650 ms               | bit-equal        |
| `q4_0`    | 372 MiB   | 1080 ms               | 595 ms                | bit-equal        |

Measurements on an Apple M4 Air, 10 ggml-cpu threads, OpenMP,
`--bench-warmup 5 --bench-runs 15`. Transcripts on both clips are
bit-equal to NeMo PyTorch reference at every tier tested, including
`q4_0`. To reproduce / compare across GGUF tiers and backends:

```bash
./build/qvac-parakeet --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --wav test/samples/sample-16k.wav \
    --bench --bench-runs 15 --bench-warmup 5 \
    --bench-json artifacts/bench/my-q8_0.json
```

For per-sub-stage encoder profiling (subsampling / CTC head / per-block
times across `n_layers = {0, 1, N/2, N}`):

```bash
./build/qvac-parakeet --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --wav test/samples/sample-16k.wav \
    --profile --profile-runs 5 --profile-warmup 2
```

**Defaults**: `q8_0` is the speed/accuracy sweet spot (2x smaller
than f16, bit-equal transcripts, 11-23 % faster than onnxruntime on
typical clips). `q4_0` is 3.5x smaller still and also bit-equal on
clean speech. `q5_0` ships but its mul_mat kernel is slower than both
on Apple Silicon, so it's only useful if you want the `q5_0` size
tier specifically.

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

**GPU Metal** — same GGUF, Metal backend (build with `-DGGML_METAL=ON`,
run with `--n-gpu-layers 1`):

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
  Transcripts            match           match
```

Summary: on CPU, onnxruntime's AMX-accelerated kernels are 12-25 %
faster than ggml-cpu. On Metal, ggml is **2.4-2.5x faster** than
onnxruntime int8 with 21x tighter variance, landing the 20 s clip's
encoder at **~73x real-time**; quant tier (f16 / Q8_0 / Q4_0) only
affects file size, not throughput, because the Metal path is
compute-bound on shader units.

## 3. Usage

### Quickstart: wav -> text

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --wav   test/samples/jfk.wav
```

Auto-routing on the model type means the same command also works on
TDT GGUFs (you get cased + punctuated text) and on Sortformer GGUFs
(you get `[start-end] speaker_N` lines instead of text). See `--help`
for the full flag set.

### Raw PCM input

For headless pipelines (ffmpeg / sox upstream, or the QVAC bindings), the
CLI also accepts raw mono PCM via `--pcm-in`. The raw stream carries no
header, so you must pass `--pcm-rate HZ` to match the model's expected
sample rate -- omitting it falls back to the model's rate with a
warning, and a mismatched rate fails fast (resampling is not yet wired):

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --pcm-in recording.raw \
    --pcm-format s16le \   # or f32le; defaults to s16le
    --pcm-rate   16000     # required for fail-fast; warning + fallback if omitted
```

### Streaming — Mode 2 (full audio in, segments streamed out)

The engine exposes three transcription entry points that mirror the qvac
SDK's `transcribe` / `transcribeStream` API:

| Entry point | Caller provides | Caller receives | Status |
|-|-|-|-|
| `Engine::transcribe()` | full audio | full text | ships |
| `Engine::transcribe_stream()` | full audio + callback | segments via callback | **ships (Mode 2)** |
| `Engine::stream_start()` -> `StreamSession` | push PCM via `feed_pcm_*()` | segments via callback | **ships (Mode 3, cache-aware inference)** |

Mode 2 runs the offline encoder once, then walks the encoder frames in
`chunk_ms`-sized windows. For CTC GGUFs it runs `ctc_greedy_decode_window`
per window and the concatenated transcript is **byte-equal** to the
non-streaming path -- `test-streaming` asserts this across chunk sizes
{250, 500, 1000, 2000, 4000, 11000} ms on every run. For TDT GGUFs it
carries `TdtDecodeState` (LSTM hidden + last token) across windows; the
non-streaming WER is preserved within `test-streaming`'s tolerance band
(40% at the most aggressive `chunk=1000 left=2000 right=500` config,
~0% at typical settings) but byte-equality with the non-streaming path
is **not** guaranteed because the joint network's emission timing can
shift slightly when the encoder context window changes.

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
  down to multiples of the encoder frame stride, which is 80 ms on
  every shipped GGUF; the implementation derives it from the model's
  mel hop length and subsampling factor).
- `--emit text` — one `[start-end] text` line per segment (default).
- `--emit jsonl` — one `{"chunk","start","end","is_final","text"}` JSON
  object per line, for easy downstream consumption.

On a 5.5 minute speech clip (`LastQuestion_long_EN.raw`, 16 kHz
s16le) Mode 2 lands at **RTF 0.046** (~22x real-time) on M4 Air with
Metal Q8_0. Segments are emitted at the `--stream-chunk-ms` cadence
once the offline encoder finishes -- Mode 2 is *cosmetic streaming*:
first segment lands after the full encoder pass.

### Streaming — Mode 3 (live duplex, cache-aware inference)

Mode 3 feeds PCM into a `StreamSession` incrementally; each chunk runs
its own encoder pass over `[left_context + chunk + right_lookahead]`
audio, slices out the center frames, and emits a segment as soon as
that chunk is processed. First segment lands at
`chunk_ms + right_lookahead_ms + encoder_time`, not after the full
utterance.

Key point: **no new model needed**. Mode 3 runs whichever
offline-trained Parakeet GGUF you have loaded (CTC or TDT) in
cache-aware inference mode. Accuracy is preserved within a few percent
of offline WER when the `left_context_ms` and `right_lookahead_ms`
budgets are reasonable
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

`--stream-left-context-ms` is the audio context per chunk (10 s is
the default; diminishing returns past 5 s); `--stream-right-lookahead-ms`
is the most impactful accuracy knob (future audio appended before
emitting). Both have `StreamingOptions` mirrors on the C++ side.

Measured on Apple M4 Air, Q8_0, Metal backend:

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

The Node binding at [qvac-lib-infer-parakeet](https://github.com/qvac/qvac-lib-infer-parakeet)
is the intended consumer for `StreamSession`; check its README for
the `qvac-parakeet.cpp` version it currently links against.

### Streaming — Sortformer (live diarization)

Phase 11.11.1 ships a push-API `SortformerStreamSession`. The session
buffers audio internally and, every `chunk_ms`, runs
`Engine::diarize()` over the trailing `history_ms` of audio, emits
segments that overlap the new chunk via callback, and slides the chunk
pointer forward.

```cpp
SortformerStreamingOptions opts;
opts.sample_rate    = 16000;
opts.chunk_ms       = 2000;     // emit cadence
opts.history_ms     = 30000;    // sliding context window
opts.threshold      = 0.5f;
opts.min_segment_ms = 200;

auto session = sortformer_engine.diarize_start(opts,
    [](const StreamingDiarizationSegment & s) {
        std::printf("[%.2f-%.2f] speaker_%d (chunk %d%s)\n",
                    s.start_s, s.end_s, s.speaker_id, s.chunk_index,
                    s.is_final ? ", final" : "");
    });

session->feed_pcm_f32(samples, n);
// ...feed more...
session->finalize();
```

CLI:

```bash
./build/qvac-parakeet \
    --model models/sortformer-4spk-v1.f16.gguf \
    --pcm-in recording.raw --pcm-format s16le \
    --stream \
    --stream-chunk-ms 2000 --stream-history-ms 30000 \
    --emit text   # or jsonl
```

Trade-offs of the Phase 11.11.1 pragmatic implementation:

- **Pro**: works with both v1 and v2 Sortformer GGUFs out of the box,
  no encoder graph split, no spkcache state. ~RTF 0.25 on M4 Air CPU
  with `chunk_ms=2000 history_ms=30000` (each chunk re-runs the full
  encoder over the trailing 30 s).
- **Pro**: speaker IDs stabilise within a few chunks once the history
  window contains both speakers' audio.
- **Con**: speaker IDs are derived from each per-chunk `diarize()`
  independently and may shift on the *very first* chunks, before the
  history window is full enough to disambiguate speakers.

Phase 11.11.2 (planned) implements true NeMo-style streaming with
`spkcache` compression + encoder graph split for fully stable
cross-chunk speaker identity at lower per-chunk compute.

### Live microphone

Three example binaries take audio from the system default mic via
[`miniaudio`](https://miniaud.io/) (single-header, MIT, vendored
under `examples/miniaudio.h`) and drive the streaming push API.
Terminal output only, no GUI. First run on macOS will prompt for
microphone access. Across all three examples: capture happens on the
audio callback thread into a mutex-guarded queue; the main thread
drains and feeds the engine, so the encoder never blocks the
capture buffer. Ctrl-C stops the device, flushes the tail, and
`finalize()`s cleanly.

#### `live-mic` -- transcription **or** diarization

`examples/live-mic.cpp` auto-detects the GGUF: a CTC/TDT model drives
`StreamSession` and prints transcript segments; a Sortformer model
drives `SortformerStreamSession` and prints `[start-end] speaker_N`
lines.

```bash
# List capture devices:
./build-metal/live-mic --list-devices

# Live transcription (Ctrl-C to stop, Metal recommended):
./build-metal/live-mic \
    --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --n-gpu-layers 1 \
    --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000

# Same, but accumulate transcript on a single line and emit a newline
# after 1 s of silence (hands-free dictation feel):
./build-metal/live-mic \
    --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
    --n-gpu-layers 1 \
    --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000 \
    --accumulate --silence-flush-ms 1000

# Live diarization (same binary, Sortformer GGUF auto-detected):
./build-metal/live-mic \
    --model models/sortformer-4spk-v1.f16.gguf \
    --chunk-ms 2000 --history-ms 30000
```

Defaults are tuned for an interactive feel: first transcription
segment lands ~2 s after you start speaking
(`chunk_ms + right_lookahead_ms + encoder_time`), then at the
`chunk_ms` cadence.

#### `live-mic-attributed` -- ASR + diarization in one binary

`examples/live-mic-attributed.cpp` loads a CTC/TDT engine and a
Sortformer engine, forwards each captured batch to both, and tags
each transcript segment with the speaker whose live diarization
range overlaps it the most:

```
[2.10-3.00] speaker_0: hello there how are you
[3.00-4.00] speaker_0: doing today
[4.00-5.20] speaker_1: I am fine thanks
```

```bash
./build/live-mic-attributed \
    --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf \
    --diar-model models/sortformer-4spk-v1.f16.gguf \
    --asr-chunk-ms 1000  --asr-left-context-ms 5000 --asr-right-lookahead-ms 1000 \
    --diar-chunk-ms 2000 --diar-history-ms 30000
```

Each captured audio batch is forwarded to both `StreamSession`
(transcription) and `SortformerStreamSession` (diarization). The
diarization callback maintains a sliding deque of recent
`[start, end, speaker]` spans; the transcription callback looks up
the most-overlapping span at the segment's time range and tags the
line. `[diar] active speaker_N at t.ts` fires on stderr at speaker
switches.

Knobs:

- `--asr-chunk-ms / --asr-left-context-ms / --asr-right-lookahead-ms`:
  same as `live-mic` for transcription.
- `--diar-chunk-ms / --diar-history-ms`: same as `Engine::diarize_start`.
- `--speaker-history-ms` (default 60000): how much diarization history
  to retain for the attribution lookup. Increase for very long
  conversations; decrease if memory is tight.
- `--asr-n-gpu-layers / --diar-n-gpu-layers`: independent GPU offload
  knobs so you can run e.g. ASR on Metal and diarization on CPU (or
  vice versa) on machines with a single GPU.
- `--accumulate`: collapse output to one line per speaker and emit
  a newline on speaker change or after `--silence-flush-ms` of
  silence (default 1000). Same UX as `live-mic --accumulate`, but
  each line is prefixed with `speaker_N:`. Output looks like:

  ```
  speaker_0: hello there how are you doing today
  speaker_1: I am fine thanks how about yourself
  speaker_0: pretty good thanks for asking
  ```

With `-DGGML_METAL=ON`, the same example runs both engines on the
GPU (use independent `--asr-n-gpu-layers` / `--diar-n-gpu-layers`
to mix CPU and GPU on a single-GPU machine):

```bash
./build-metal/live-mic-attributed \
    --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf  --asr-n-gpu-layers 1 \
    --diar-model models/sortformer-4spk-v1.f16.gguf    --diar-n-gpu-layers 1 \
    --accumulate
```

The same Phase 11.11.1 sliding-history caveat from "Streaming --
Sortformer" applies to the speaker IDs the attribution layer sees.

## 4. Optional: validate against NeMo PyTorch

CTC parity (mel + encoder + greedy decode):

```bash
python scripts/dump-ctc-reference.py \
    --wav test/samples/jfk.wav \
    --out artifacts/ctc-ref

./build/test-mel     models/parakeet-ctc-0.6b.gguf test/samples/jfk.wav artifacts/ctc-ref/mel.npy
./build/test-encoder models/parakeet-ctc-0.6b.gguf artifacts/ctc-ref
./build/test-ctc     models/parakeet-ctc-0.6b.gguf artifacts/ctc-ref/logits.npy
```

TDT parity (encoder per-stage; the decoder is checked end-to-end via
the CLI transcript byte-equality check on `jfk.wav`):

```bash
python scripts/dump-tdt-reference.py \
    --wav test/samples/jfk.wav \
    --out artifacts/tdt-ref

./build/test-tdt-encoder-parity \
    models/parakeet-tdt-0.6b-v3.q8_0.gguf test/samples/jfk.wav artifacts/tdt-ref
```

Sortformer parity (mel + encoder + speaker-prob head):

```bash
python scripts/dump-sortformer-reference.py \
    --wav  test/samples/two-speakers-16k.wav \
    --out  artifacts/sortformer-ref

./build/test-sortformer-parity \
    models/sortformer-4spk-v1.f16.gguf test/samples/two-speakers-16k.wav artifacts/sortformer-ref
```

Streaming smoke tests (Mode 1/2/3 byte-equality + WER tolerance for
CTC/TDT; sliding-history push API + no-duplicate + single-`is_final`
for Sortformer):

```bash
./build/test-streaming \
    --model models/parakeet-ctc-0.6b.q8_0.gguf --wav test/samples/jfk.wav

./build/test-sortformer-streaming \
    --model models/sortformer-4spk-v1.f16.gguf --wav test/samples/two-speakers-16k.wav
```

Expected per-stage rel error (NeMo PyTorch vs C++ at `--quant f16`):

```
Stage A  log_mel               ~ 1e-4 inner / ~ 2e-3 boundary (f32 FFT)
Stage B  subsampling_out       rel ~ 1e-3 (f16 quantization floor)
Stage C  block_0_out           rel ~ 1e-3
Stage D  block_last_out        rel ~ 2e-3
Stage E  ctc_logits            rel ~ 1e-3   (CTC head only)
Stage F  decoded transcript    edit distance = 0 on clean speech
Stage S  speaker_probs         rel ~ 2e-4   (Sortformer head)
```

At `--quant q8_0` through `q4_0` the per-stage rel inflates by ~3x
to ~25x, but the CTC transcript stays bit-equal on clean speech. See
`PROGRESS.md` §5.12 for the CTC quant sweep, §10.x for TDT, and §11.x
for Sortformer.

## Current status

Phases 0 through 11 have shipped (see `PROGRESS.md` for the full
journal). Outstanding workstreams are Phase 8.5 (true KV cache + conv
state for ~6x compute reduction on long-form Mode 3 audio) and Phase
11.11.2 (NeMo-style spkcache + encoder graph split for fully stable
Sortformer streaming speaker IDs).

Headline highlights (per phase, one bullet each; PROGRESS.md `§N.x`
has the full round-by-round journal):

- **Parity (Phase 4)**: `qvac-parakeet --model ... --wav ...` produces
  the expected transcript end-to-end, matching NeMo PyTorch
  bit-equivalently on `jfk.wav` and `sample-16k.wav` at every quant
  tier (f16 through Q4_0) on both CPU and Metal backends. Per-stage
  numerical parity at the f16 quantization floor (~1-2e-3 rel vs NeMo
  PyTorch) on every intermediate encoder tensor.
- **CPU optimisation (Phase 5)**: encoder runs 22x real-time on an
  M4 Air CPU at Q8_0 — 12 % faster than ONNX f16, 22 % slower than
  ONNX int8.
- **Metal (Phase 6)**: encoder runs 73x real-time on the M4 Air GPU
  at Q8_0 — 2.5x faster than onnxruntime int8 with 21x tighter
  variance (0.83 ms stdev).
- **Mode 2 streaming (Phase 7)**: `Engine::transcribe_stream()` runs
  the offline encoder once, walks the encoder frames in `chunk_ms`
  windows, emits per-segment callbacks. Byte-equal to non-streaming
  on CTC; WER-bounded on TDT.
- **Mode 3 live duplex (Phase 8)**: `Engine::stream_start()` ->
  `StreamSession` push API. Cache-aware inference on the existing
  offline GGUF (CTC or TDT); no new checkpoint needed. ~4 % WER on a
  5.5 min sci-fi clip with ~4 s first-segment latency at defaults.
- **Multi-model loader (Phase 9)**: same converter + same `Engine`
  handle the 1.1B variants alongside the 0.6B baselines.
- **TDT decoder (Phase 10)**: `nvidia/parakeet-tdt-0.6b-v3` and
  `parakeet-tdt-1.1b` ported -- multilingual transcription with
  punctuation + capitalisation. 2-layer LSTM prediction + joint MLP
  + transducer greedy decode (CPU, f32 after dequant). Mode 1, 2 and
  3 all support TDT GGUFs.
- **Sortformer diarization (Phase 11)**: `diar_sortformer_4spk-v1`
  and `diar_streaming_sortformer_4spk-v2` ported -- 4-speaker
  diarization with rel 2.0e-4 vs NeMo on speaker probabilities.
  `Engine::diarize()` API + CLI auto-routing. §11.10 ships
  `transcribe_with_speakers` for combined ASR + speaker attribution.
  §11.11.1 ships `Engine::diarize_start()` ->
  `SortformerStreamSession` for live diarization (sliding-history v1;
  Phase 11.11.2 NeMo-style spkcache streaming pending).

Next: Phase 8.5 (true KV cache + conv state for ~6x compute reduction
on long-form Mode 3 audio without accuracy change), Accelerate BLAS
for the TDT decoder's LSTM + joint gemvs and Sortformer's transformer
attention, `CONV_2D_DW` on Metal (upstream ggml contribution), Metal
flash-attn, Phase 11.11.2 Sortformer streaming, EOU pipelines.

## Repository layout

```
qvac-parakeet.cpp/
  ggml/                          pristine ggml clone (not tracked; populated
                                   by scripts/setup-ggml.sh, or skipped entirely
                                   when building with -DQVAC_PARAKEET_USE_SYSTEM_GGML=ON)
  src/
    main.cpp                     CLI (wav / raw PCM -> text or speaker segments,
                                   + Mode 2/3 transcription streaming, sliding-history
                                   diarization streaming, attribution) + qvac_parakeet_cli_main
                                   + transcribe_wav (CTC-only one-shot helper)
    cli_main.cpp                 thin main() -> qvac_parakeet_cli_main shim
    parakeet_ctc.{h,cpp}         GGUF loader + FastConformer encoder ggml graph
                                   + CTC head + greedy decode (shared by all engines;
                                   model_type field selects the decoder)
    parakeet_tdt.{h,cpp}         TDT decoder: 2-layer LSTM prediction + joint MLP
                                   + transducer greedy decode (CPU)
    parakeet_sortformer.{h,cpp}  Sortformer diarization: encoder_proj + 18-layer
                                   Transformer encoder + ReLU MLP + sigmoid head + segmenter
    parakeet_engine.cpp          Engine + StreamSession + SortformerStreamSession
                                   (transcribe, transcribe_stream, stream_start, diarize,
                                    diarize_start, transcribe_with_speakers)
    mel_preprocess.{h,cpp}       wav I/O + STFT + mel + CMVN
    sentencepiece_bpe.{h,cpp}    SentencePiece BPE detokenizer (CTC + TDT)
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare
    test_*.cpp                   per-stage numerical-parity harnesses (mel, encoder,
                                   ctc, tdt-encoder, sortformer) + streaming
                                   validation (test-streaming, test-sortformer-streaming)
  include/qvac-parakeet/
    qvac-parakeet.h              CLI entry (qvac_parakeet_cli_main) + library overview
    ctc/engine.h                 persistent multi-engine Engine umbrella + StreamSession +
                                   SortformerStreamSession + transcribe_with_speakers.
                                   The header path "ctc/" is historical -- the API now
                                   covers CTC, TDT and Sortformer GGUFs.
    ctc/pipeline.h               one-shot wav -> text API (CTC GGUFs only;
                                   hard-errors on TDT/Sortformer)
  examples/
    live-mic.cpp                 live microphone -> transcription (CTC/TDT) or live
                                   diarization (Sortformer); auto-detects the GGUF.
    live-mic-attributed.cpp      live microphone -> dual-engine ASR + Sortformer
                                   with per-segment speaker attribution.
    miniaudio.h                  vendored single-header audio capture (MIT).
  scripts/
    setup-ggml.sh                pin + clone ggml
    convert-parakeet-ctc-to-gguf.py    .nemo -> GGUF (auto-detects CTC / TDT / Sortformer)
    dump-ctc-reference.py        NeMo PyTorch -> .npy reference tensors (CTC stages)
    dump-tdt-reference.py        NeMo PyTorch -> .npy reference tensors (TDT stages)
    dump-sortformer-reference.py NeMo PyTorch -> .npy reference tensors (Sortformer stages)
    dump-block0-substages.py     per-sub-stage timing inputs for --profile
    ref-encoder-from-gguf.py     run the GGUF encoder in PyTorch as a parity oracle
    streaming-reference.py       reference per-chunk outputs for streaming validation
    verify-gguf-roundtrip.py     load a GGUF and assert all expected tensors are present
    quantize-ctc-onnx-int8.py    int8-quantize an ONNX CTC export (for the Node binding)
    download-all-models.sh       pre-fetch every supported .nemo (and ONNX bundle)
    transcribe.sh                wav -> text wrapper
  cmake/                         CMake package config (for vcpkg follow-up)
  test/samples/                  fixture wavs (jfk.wav, sample-16k.wav)
  artifacts/                     dumped reference tensors (.npy) per engine; not tracked
  models/                        downloaded .nemo + converted .gguf checkpoints; not tracked
  PROGRESS.md                    chronological development journal
  README.md                      this file
```

## License

Released under the [Apache License 2.0](LICENSE).

**Model licenses**: every NVIDIA Parakeet (CTC, TDT) and Sortformer
checkpoint listed in the model table at the top of this README ships
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) on
Hugging Face -- check each model card for the canonical attribution.
This repository only ships the inference code; model weights are
downloaded on demand by the converter / `download-all-models.sh`.

The bundled `ggml/` is MIT-licensed (see `ggml/LICENSE`).
