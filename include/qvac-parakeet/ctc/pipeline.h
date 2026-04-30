#pragma once

// One-shot Parakeet-CTC transcription: wav -> text.
//
// CTC GGUFs only. This entry point preprocesses the wav into a log-mel
// spectrogram using the filterbank baked into the GGUF, runs the
// FastConformer encoder + CTC head, and greedily decodes with
// collapse-repeats + blank-removal + SentencePiece detokenization.
//
// Feeding a TDT, EOU or Sortformer GGUF here is a hard error (return
// code 11) -- those checkpoints have no CTC head and require different
// decoders. For multi-engine support (CTC + TDT + EOU + Sortformer
// behind one API), use the persistent `Engine` umbrella in
// <qvac-parakeet/ctc/engine.h>, which auto-detects the model type at
// load and dispatches accordingly.
//
// `Engine` is also preferred for any persistent / many-utterance use even
// on CTC GGUFs because it amortises the model load cost.
//
// Implementation in src/parakeet_pipeline.cpp.

#include "../api.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet {

struct TranscribeOptions {
    std::string model_gguf_path;
    std::string wav_path;

    int n_threads = 0;
    int n_gpu_layers = 0;

    bool verbose = false;
};

struct TranscribeResult {
    std::string text;
    std::vector<int32_t> token_ids;

    double preprocess_ms = 0.0;
    double encoder_ms    = 0.0;
    double decode_ms     = 0.0;
    double total_ms      = 0.0;

    int audio_samples    = 0;
    int sample_rate      = 16000;
    int mel_frames       = 0;
    int encoder_frames   = 0;
};

QVAC_PARAKEET_API int transcribe_wav(const TranscribeOptions & opts,
                                     TranscribeResult       & result);

namespace ctc {
    using namespace ::qvac_parakeet;
}

}
