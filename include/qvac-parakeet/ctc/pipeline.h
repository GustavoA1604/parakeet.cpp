#pragma once

// One-shot Parakeet-CTC transcription: wav -> text.
//
// Loads the GGUF, preprocesses the wav into an 80-channel log-mel
// spectrogram using the filterbank baked into the GGUF, runs the
// FastConformer encoder + CTC head, and greedily decodes with
// collapse-repeats + blank-removal + SentencePiece detokenization.
//
// For persistent use (many utterances on the same model), prefer
// <qvac-parakeet/ctc/engine.h>, which amortises the model load cost.
//
// Implementation in src/parakeet_ctc.cpp.

#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet::ctc {

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

int transcribe_wav(const TranscribeOptions & opts, TranscribeResult & result);

}
