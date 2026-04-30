#pragma once

// Top-level QVAC Parakeet library entry points.
//
// The library ships four engine families behind a single `Engine` umbrella
// in <qvac-parakeet/ctc/engine.h> (the header path is historical -- the
// Engine accepts CTC, TDT, EOU and Sortformer GGUFs alike and auto-detects
// the model type at load time):
//
//   - Parakeet-CTC 0.6B / 1.1B  -- English transcription
//   - Parakeet-TDT 0.6B-v3 / 1.1B -- multilingual transcription with
//     punctuation and capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Parakeet-EOU 120M (`parakeet_realtime_eou_120m-v1`) -- low-latency
//     streaming ASR with native `<EOU>` end-of-utterance token
//   - Sortformer 4spk v1 / v2 -- 4-speaker diarization (offline; v2 also
//     has Phase 11.11.1 sliding-history live streaming)
//
// Combined ASR + Sortformer "who said what" is exposed via the free
// function `transcribe_with_speakers(...)`.
//
// Three layers of API:
//
//   1. CLI dispatcher: `qvac_parakeet_cli_main(argc, argv)` -- same flags
//      as the `qvac-parakeet` binary.
//   2. Persistent `Engine` (load model once, run many): see
//      <qvac-parakeet/ctc/engine.h>.
//   3. One-shot wav->text helper for CTC GGUFs only:
//      <qvac-parakeet/ctc/pipeline.h>. Feeding a TDT, EOU or Sortformer
//      GGUF here returns `11` (those checkpoints have no CTC head and
//      require different decoders); use `Engine` for those.

#include "api.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

QVAC_PARAKEET_API int qvac_parakeet_cli_main(int argc, char ** argv);

// Install a host-supplied log sink. Mirrors `llama_log_set` from
// the qvac-fabric-llm.cpp sibling so addons can demux every
// (level, text) emission from libqvac-parakeet into their own
// structured logging without scraping stderr.
//
// Pass `cb = nullptr` to revert to the default sink (stderr). The
// signature is `ggml_log_callback`, the same one `ggml_log_set`
// already accepts, so a single host shim can fan one log pipe out
// across both libraries. Default is no callback installed (stderr
// fallback), matching pre-shim behaviour byte-for-byte.
QVAC_PARAKEET_API void qvac_parakeet_log_set(ggml_log_callback cb,
                                             void *            user_data);

#ifdef __cplusplus
}
#endif
