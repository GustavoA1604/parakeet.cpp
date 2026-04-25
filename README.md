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

The TDT decoder (prediction net + joint net + transducer greedy) runs
on CPU in pure float32 after dequantizing its ~70 MiB of weights once
at Engine construction. All three entry points work for both model
families:

- `Engine::transcribe()` (one-shot) — CTC or TDT.
- `Engine::transcribe_stream()` (Mode 2, offline encoder + streamed
  segments) — CTC or TDT.
- `Engine::stream_start()` -> `StreamSession` (Mode 3, live duplex
  cache-aware) — CTC or TDT; TDT needs slightly more context than CTC
  at the same chunk size (TDT's transducer is more sensitive to
  missing right-lookahead at chunk boundaries; typical WER delta
  vs offline is +5-10 %).

Mirrors [`chatterbox.cpp`](https://github.com/GustavoA1604/chatterbox.cpp)'s
layout and staged-validation methodology, so contributors familiar with
that repo will find the same file structure here.

---

## Pipeline at a glance

```
   16 kHz mono wav                                                  output
          |                                                           ^
          v                                                           |
  +-----------------------------------------------------------------+
  |                            qvac-parakeet                        |
  |                                                                 |
  |    wav  ->  log-mel (80 or 128)  ->  FastConformer encoder      |
  |             (STFT + CMVN)            (subsampling 8x, 17-42     |
  |                                       conformer blocks)         |
  |                                                                 |
  |    decoder dispatched on GGUF metadata:                         |
  |      CTC        head + greedy decode      -> text               |
  |      TDT        LSTM pred + joint MLP +   -> text +             |
  |                 transducer greedy            punctuation        |
  |      Sortformer encoder_proj + 18L TF +   -> {speaker, t0, t1}* |
  |                 sigmoid head                                    |
  +-----------------------------------------------------------------+
          ^                                                           |
          |                                                           v
   dr_wav / miniaudio                                  SentencePiece BPE
                                                       (CTC/TDT only;
                                                        embedded in GGUF)
```

Each `.gguf` ships everything its decoder needs in a single file
(encoder weights, decoder weights, precomputed mel filterbank, and the
SentencePiece tokenizer where applicable). The same C++ `Engine`
auto-detects the model type (CTC / TDT / Sortformer) at load time and
dispatches to the right decoder, so the public API is single-engine
from the consumer's perspective.

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
    --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --wav   test/samples/jfk.wav
```

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
filterbank, depthwise/ small 2D convs) always stay at f32/f16.

The block-quantised formats (`q8_0`, `q5_0`, `q4_0`) require the
last-dim of each tensor to be a multiple of the block size (32 for
all three). Tensors whose `shape[-1] % 32 != 0` are silently kept at
f16 by the converter; this is why a `q4_0` GGUF lands at 372 MiB
rather than the theoretical 4-bit minimum -- the un-quantisable
fragments stay at f16. Practically every "fat" 2D matrix in the
shipped models meets the alignment, so the headline size is close to
the theoretical floor.

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
is the intended consumer for `StreamSession` -- the push API is
designed so each incoming `Buffer` from the binding's existing
`append({type:'audio', data})` flow maps to `feed_pcm_i16` and
`{type:'end of job'}` maps to `finalize()`. Cross-check the binding's
README for which version of `qvac-parakeet.cpp` it currently links
against.

### Live microphone example

`examples/live-mic.cpp` wraps `StreamSession` around
[`miniaudio`](https://miniaud.io/) (single-header, MIT, vendored under
`examples/miniaudio.h`) for real-time transcription from the default
capture device on macOS / Linux / Windows. Terminal output only, no GUI.

```bash
# Built as part of the default CLI target set when the project is the
# top-level CMake (`QVAC_PARAKEET_BUILD_EXAMPLES` defaults to
# `QVAC_PARAKEET_STANDALONE_DEFAULT`, i.e. ON for `cmake -S . -B build`
# but OFF for sub-projects). Pass `-DQVAC_PARAKEET_BUILD_EXAMPLES=ON`
# explicitly when consuming this repo as a sub-project.

# List capture devices:
./build-metal/live-mic --list-devices

# Transcribe live (Ctrl-C to stop, Metal backend recommended):
./build-metal/live-mic \
    --model models/parakeet-ctc-0.6b.q8_0.gguf \
    --n-gpu-layers 1 \
    --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000

# Same, but accumulate transcript on a single line and only emit a
# newline after 1 s of silence (hands-free dictation feel):
./build-metal/live-mic \
    --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
    --n-gpu-layers 1 \
    --chunk-ms 1000 --left-context-ms 5000 --right-lookahead-ms 1000 \
    --accumulate --silence-flush-ms 1000
```

First time you run it macOS will prompt for microphone access. The
capture thread pushes f32 samples into a mutex-guarded queue; the main
thread drains the queue and calls `StreamSession::feed_pcm_f32`, so
the encoder runs off the audio callback thread (no capture-buffer
stalls). Ctrl-C sets a stop flag, the capture device is stopped, the
tail buffer is flushed, `finalize()` emits the last segment, and the
binary exits cleanly.

Defaults chosen for an interactive feel: first segment lands ~2 s
after you start speaking
(`chunk_ms + right_lookahead_ms + encoder_time`); segments afterward
at the `chunk_ms` cadence.

When `--model` points at a Sortformer GGUF (e.g.
`models/sortformer-4spk-v1.f16.gguf`) `live-mic` automatically switches
to live diarization mode: instead of transcript segments it prints
`[start-end] speaker_N` per chunk via the same push API
(`SortformerStreamSession`). See "Streaming — Sortformer (live
diarization)" below.

### Live microphone with speaker attribution (`live-mic-attributed`)

`examples/live-mic-attributed.cpp` runs a transcription engine
(CTC/TDT) and a Sortformer engine on the **same** mic feed and tags
each transcript segment with the speaker whose live diarization range
overlaps it the most. Output:

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
`[start, end, speaker]` spans; the transcription callback looks up the
most-overlapping span at the segment's time range and tags the line.
A short stderr log line `[diar] active speaker_N at t.ts` fires on
speaker switches.

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
- `--accumulate`: instead of one line per transcription chunk,
  accumulate text on a single line per speaker and emit a newline on
  speaker change or after `--silence-flush-ms` of silence (default
  1000). Same UX as `live-mic --accumulate`, but each line is
  prefixed with `speaker_N:`. Output looks like:

  ```
  speaker_0: hello there how are you doing today
  speaker_1: I am fine thanks how about yourself
  speaker_0: pretty good thanks for asking
  ```

Metal-backed binary lives under `build-metal/live-mic-attributed`
once the project has been configured with `-DGGML_METAL=ON` (the
existing `build-metal/` directory in this repo is already configured
for that):

```bash
./build-metal/live-mic-attributed \
    --asr-model  models/parakeet-tdt-0.6b-v3.q8_0.gguf  --asr-n-gpu-layers 1 \
    --diar-model models/sortformer-4spk-v1.f16.gguf    --diar-n-gpu-layers 1 \
    --accumulate
```

Caveat (Phase 11.11.1): the underlying `SortformerStreamSession` uses
sliding-history streaming, so speaker IDs may shift in the very first
chunks before the history fills. Phase 11.11.2 will fix this with
spkcache compression.

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
  no encoder graph split, no spkcache state. ~RTF 0.25 on M3 with
  `chunk_ms=2000 history_ms=30000` (each chunk re-runs the full
  encoder over the trailing 30 s).
- **Pro**: speaker IDs stabilise within a few chunks once the history
  window contains both speakers' audio.
- **Con**: speaker IDs are derived from each per-chunk `diarize()`
  independently and may shift on the *very first* chunks, before the
  history window is full enough to disambiguate speakers.

Phase 11.11.2 (planned) implements true NeMo-style streaming with
`spkcache` compression + encoder graph split for fully stable
cross-chunk speaker identity at lower per-chunk compute.

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

Phases 0 through 11 have shipped (see `PROGRESS.md` for the full
journal). Outstanding workstreams are Phase 8.5 (true KV cache + conv
state for ~6x compute reduction on long-form Mode 3 audio) and Phase
11.11.2 (NeMo-style spkcache + encoder graph split for fully stable
Sortformer streaming speaker IDs).

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
  es/fr/de/it/pt/ru samples. One-shot + Mode 2 + Mode 3 streaming
  all work with TDT GGUFs (phase 10.5), including `live-mic` for
  native microphone capture.
- **Phase 11 — Sortformer (4-speaker diarization)**: port of
  `nvidia/diar_sortformer_4spk-v1` — 18-layer FastConformer encoder
  (reused) -> Linear projection (512 -> 192) -> 18-layer post-LN
  Transformer encoder -> ReLU MLP -> sigmoid head producing per-frame
  speaker probabilities. New `Engine::diarize()` API + CLI
  auto-routing. Output: per-frame probabilities and threshold-based
  segments {speaker, start, end}. Speaker probability parity is
  rel 2.0e-4 vs NeMo reference.
  - **§11.10 speaker-attributed transcription** ships:
    `transcribe_with_speakers(sortformer_engine, asr_engine, ...)`
    plus CLI `--diarization-model PATH`. Combines Sortformer
    segments with CTC/TDT transcripts in one C++ binary. Same
    pipeline as the qvac binding's `quickstart-diarized.js`, but
    native.
  - **§11.11.1 Sortformer live streaming (pragmatic v1)** ships:
    `Engine::diarize_start()` -> `SortformerStreamSession` with
    `feed_pcm_f32/i16` + `finalize()` push API. Sliding-history
    implementation: each `chunk_ms` re-runs `diarize()` over the
    trailing `history_ms` and emits segments overlapping the new
    chunk. Auto-routed by the CLI when `--stream` is set on a
    Sortformer model. The `live-mic` example also auto-detects
    Sortformer GGUFs and switches to live diarization. Trade-off:
    speaker IDs may shift in the very first chunks until history
    fills; stable thereafter.
  - **§11.11.0 Sortformer v2 offline support** ships:
    `nvidia/diar_streaming_sortformer_4spk-v2` GGUF converts and
    runs through the same offline `diarize()` path. Live duplex
    API (chunked attention + spkcache + FIFO state machine) is the
    next streaming-diarization workstream.

Next: Phase 8.5 (true KV cache + conv state for ~6x compute reduction on
long-form audio without accuracy change), Accelerate BLAS for the TDT
decoder's LSTM + joint gemvs and Sortformer's transformer attention,
`CONV_2D_DW` on Metal (upstream ggml contribution), Metal flash-attn,
Sortformer v2 live streaming, EOU pipelines.

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

**Model license**: the Parakeet-CTC-0.6B weights are licensed
[CC-BY-4.0 by NVIDIA](https://huggingface.co/nvidia/parakeet-ctc-0.6b).
This repository only ships the inference code; model weights are
downloaded on demand.

The bundled `ggml/` is MIT-licensed (see `ggml/LICENSE`).
