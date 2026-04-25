#pragma once

// Top-level QVAC Parakeet library entry points.
//
// The library ships four engine families behind a single `Engine` umbrella
// in <qvac-parakeet/ctc/engine.h> (the header path is historical -- the
// Engine accepts CTC, TDT and Sortformer GGUFs alike and auto-detects the
// model type at load time):
//
//   - Parakeet-CTC 0.6B / 1.1B  -- English transcription
//   - Parakeet-TDT 0.6B-v3 / 1.1B -- multilingual transcription with
//     punctuation and capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Sortformer 4spk v1 / v2 -- 4-speaker diarization (offline; v2 also
//     has Phase 11.11.1 sliding-history live streaming)
//   - Combined ASR + Sortformer "who said what" via the free function
//     `transcribe_with_speakers(...)`
//
// Three layers of API:
//
//   1. CLI dispatcher: `qvac_parakeet_cli_main(argc, argv)` -- same flags
//      as the `qvac-parakeet` binary.
//   2. Persistent `Engine` (load model once, run many): see
//      <qvac-parakeet/ctc/engine.h>.
//   3. One-shot wav->text helper for CTC GGUFs only:
//      <qvac-parakeet/ctc/pipeline.h> (TDT and Sortformer GGUFs throw
//      from this entry point; use `Engine` for those).

#ifdef __cplusplus
extern "C" {
#endif

int qvac_parakeet_cli_main(int argc, char ** argv);

#ifdef __cplusplus
}
#endif
