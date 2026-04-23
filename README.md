# qvac-parakeet.cpp

**Parakeet-CTC-0.6B** (NVIDIA, CC-BY-4.0 FastConformer ASR model) ported to
[`ggml`](https://github.com/ggml-org/ggml). Pure C++/ggml inference on CPU
(GPU backends and TDT / EOU / Sortformer pipelines land as follow-ups),
with no runtime dependency on Python, PyTorch, or onnxruntime.

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

This produces the main binary plus per-stage validation harnesses:

| Binary                  | What it does |
|-------------------------|--------------|
| `build/qvac-parakeet`   | End-to-end: wav -> text (FastConformer + CTC greedy decode + SentencePiece detokenize). |
| `build/test-mel`        | 16 kHz 80-ch log-mel parity vs NeMo `AudioToMelSpectrogramPreprocessor`. |
| `build/test-encoder`    | FastConformer encoder per-stage parity vs `dump-ctc-reference.py`. |
| `build/test-ctc`        | CTC head + greedy decode parity vs NeMo `transcribe()`. |

## 2. One-time: convert weights

```bash
python -m venv venv && . venv/bin/activate
pip install "nemo_toolkit[asr]" gguf numpy soundfile librosa sentencepiece

python scripts/convert-parakeet-ctc-to-gguf.py \
  --ckpt models/parakeet-ctc-0.6b.nemo \
  --out  models/parakeet-ctc-0.6b.gguf
```

The script downloads `nvidia/parakeet-ctc-0.6b` from Hugging Face on
first run if the local path doesn't exist. The SentencePiece tokenizer
(`tokenizer.model`) and the precomputed mel filterbank are embedded
directly into the GGUF as standard `tokenizer.ggml.*` metadata and a
named `preproc/mel_filterbank` tensor, so the C++ binary is
self-contained.

## 3. Run - wav -> text

```bash
./build/qvac-parakeet \
    --model models/parakeet-ctc-0.6b.gguf \
    --wav   test/samples/jfk.wav
```

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

Expected per-stage rel error targets (NeMo vs C++):

```
Stage A  log_mel               rel ~ 1e-4 (inner) / ~ 2e-3 (boundary; f32 FFT)
Stage B  subsampling_out       target rel < 1e-4   (phase 3)
Stage C  block_0_out           target rel < 1e-4   (phase 3)
Stage D  block_23_out          target rel < 1e-4   (phase 3)
Stage E  ctc_logits            target rel < 2e-4   (phase 4)
Stage F  decoded transcript    target edit distance = 0 on clean speech
```

## Current status

Phases 0 through 4 are complete: `qvac-parakeet --model ... --wav ...`
produces the expected transcript end-to-end, matching NeMo
bit-equivalently on the greedy-decoded text on `jfk.wav`.  Per-stage
numerical parity is at the f16 quantization floor (1–2e-3 rel vs NeMo
PyTorch) on every intermediate tensor.  Benchmark on Apple Silicon CPU:
11 s of audio in ~1.05 s (RTF 0.10) on an unoptimized single-core
build.  Phase 5 (CPU optimization pass) is the remaining work.

## Repository layout

```
qvac-parakeet.cpp/
  ggml/                          pristine ggml clone (not tracked; populated
                                   by scripts/setup-ggml.sh, or skipped entirely
                                   when building with -DQVAC_PARAKEET_USE_SYSTEM_GGML=ON)
  src/
    main.cpp                     CLI (wav -> text) + qvac_parakeet_cli_main impl
    cli_main.cpp                 thin main() -> qvac_parakeet_cli_main shim
    parakeet_ctc.{h,cpp}         FastConformer encoder + CTC head ggml graph + GGUF loader
    mel_preprocess.{h,cpp}       wav I/O + STFT + mel + CMVN
    sentencepiece_bpe.{h,cpp}    SentencePiece BPE detokenizer
    dr_wav.h                     vendored single-header WAV reader
    npy.h                        minimal .npy load / save + compare
    test_*.cpp                   per-stage numerical-parity harnesses
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
