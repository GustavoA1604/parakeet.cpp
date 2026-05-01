#pragma once

// Top-level QVAC Parakeet library aggregator.
//
// Single-include convenience header; each per-concern header is also
// usable directly when consumers want to be selective:
//
//   <qvac-parakeet/export.h>      - QVAC_PARAKEET_API visibility macro
//   <qvac-parakeet/cli.h>         - qvac_parakeet_cli_main C entry point
//   <qvac-parakeet/log.h>         - qvac_parakeet_log_set host log sink
//   <qvac-parakeet/engine.h>      - Engine + EngineOptions / EngineResult
//                                   (CTC, TDT, EOU, Sortformer behind one
//                                   class)
//   <qvac-parakeet/streaming.h>   - StreamingOptions / StreamingSegment /
//                                   StreamSession + cross-engine
//                                   StreamEvent + VadState +
//                                   StreamEventType
//   <qvac-parakeet/diarization.h> - DiarizationOptions / Result +
//                                   SortformerStreamingOptions /
//                                   SortformerStreamSession
//   <qvac-parakeet/attributed.h>  - transcribe_with_speakers + the
//                                   attributed-segment types it emits
//
// Engine families behind the umbrella `Engine` (auto-routed by GGUF
// metadata at load time):
//
//   - Parakeet-CTC 0.6B / 1.1B  -- English transcription
//   - Parakeet-TDT 0.6B-v3 / 1.1B -- multilingual transcription with
//     punctuation and capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Parakeet-EOU 120M (`parakeet_realtime_eou_120m-v1`) -- low-latency
//     streaming ASR with native `<EOU>` end-of-utterance token
//   - Sortformer 4spk v1 / v2 -- 4-speaker diarization (offline; v2 also
//     has Phase 11.11.1 sliding-history live streaming)

#include "export.h"
#include "cli.h"
#include "log.h"
#include "engine.h"
#include "streaming.h"
#include "diarization.h"
#include "attributed.h"
