#pragma once

// Persistent Parakeet-CTC engine.
//
// Loads the GGUF once and keeps the preprocessor filterbank + encoder
// weights + CTC head + SentencePiece vocab + backend resident so that
// subsequent calls to `transcribe()` pay only the mel extraction +
// encoder forward + CTC decode cost.
//
// Usage:
//
//     using qvac_parakeet::ctc::Engine;
//     using qvac_parakeet::ctc::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/parakeet-ctc-0.6b.gguf";
//     opts.n_threads       = 8;
//
//     Engine engine(opts);
//     for (const auto & wav_path : wavs) {
//         auto result = engine.transcribe(wav_path);
//         std::puts(result.text.c_str());
//     }
//
// Not thread-safe for concurrent `transcribe()` calls on the same
// instance (the encoder's graph allocator is shared state).  `cancel()`
// is safe to call from any thread.
//
// Implementation in src/parakeet_ctc.cpp.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qvac_parakeet::ctc {

struct EngineOptions {
    std::string model_gguf_path;

    int n_gpu_layers = 0;
    int n_threads    = 0;

    bool verbose     = false;
};

struct EngineResult {
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

class Engine {
public:
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    EngineResult transcribe(const std::string & wav_path);

    EngineResult transcribe_samples(const float * samples,
                                    int n_samples,
                                    int sample_rate);

    void cancel();

    const EngineOptions & options() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}
