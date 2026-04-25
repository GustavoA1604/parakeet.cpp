#pragma once

// Persistent Parakeet-CTC engine.
//
// Loads the GGUF once and keeps the preprocessor filterbank + encoder
// weights + CTC head + SentencePiece vocab + backend resident so that
// subsequent calls to `transcribe()` pay only the mel extraction +
// encoder forward + CTC decode cost.
//
// Three entry points mirror the qvac/packages/sdk transcription API:
//
//   1. transcribe()                - full audio in, full text out (one-shot).
//   2. transcribe_stream()         - full audio in up front, segments streamed
//                                    out via callback as they're produced.
//                                    Runs the offline encoder once, then walks
//                                    CTC frames in chunk_ms-sized windows.
//                                    Zero accuracy delta vs transcribe().
//   3. stream_start() -> StreamSession
//                                  - true duplex: caller pushes PCM over time
//                                    via feed_pcm_*, finalize() emits the tail.
//                                    Requires a cache-aware streaming GGUF;
//                                    errors on today's offline GGUF until the
//                                    Phase 2 streaming pipeline lands.
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
// Implementation in src/parakeet_engine.cpp.

#include <cstdint>
#include <functional>
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

struct StreamingOptions {
    int sample_rate  = 16000;
    int chunk_ms     = 1000;

    int left_context_ms    = 10000;
    int right_lookahead_ms = 2000;

    bool emit_partials = false;
};

struct DiarizationOptions {
    float threshold      = 0.5f;
    int   min_segment_ms = 0;
};

struct DiarizationSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

struct StreamingDiarizationSegment {
    int    speaker_id  = 0;
    double start_s     = 0.0;
    double end_s       = 0.0;
    int    chunk_index = 0;
    bool   is_final    = false;
};

struct SortformerStreamingOptions {
    int   sample_rate     = 16000;

    int   chunk_ms        = 2000;
    int   history_ms      = 30000;

    float threshold       = 0.5f;
    int   min_segment_ms  = 200;

    bool  emit_partials   = true;
};

using SortformerSegmentCallback =
    std::function<void(const StreamingDiarizationSegment &)>;

struct DiarizationResult {
    std::vector<DiarizationSegment> segments;
    std::vector<float> speaker_probs;
    int    n_frames        = 0;
    int    num_spks        = 0;
    double frame_stride_s  = 0.08;

    int    audio_samples   = 0;
    int    sample_rate     = 16000;
    double preprocess_ms   = 0.0;
    double encoder_ms      = 0.0;
    double decode_ms       = 0.0;
    double total_ms        = 0.0;
};

struct AttributedSegment {
    int         speaker_id = 0;
    std::string text;
    double      start_s    = 0.0;
    double      end_s      = 0.0;
};

struct AttributedTranscriptionOptions {
    DiarizationOptions diarization;
    bool   merge_same_speaker  = true;
    int    min_segment_ms      = 200;
    int    pad_segment_ms      = 0;
};

struct AttributedTranscriptionResult {
    std::vector<AttributedSegment> segments;
    DiarizationResult              diarization;
    int                            asr_calls    = 0;
    double                         total_ms     = 0.0;
    int                            audio_samples = 0;
    int                            sample_rate   = 16000;
};

struct StreamingSegment {
    std::string text;
    std::vector<int32_t> token_ids;

    double start_s = 0.0;
    double end_s   = 0.0;

    int  chunk_index = 0;
    bool is_final    = true;

    double encoder_ms = 0.0;
    double decode_ms  = 0.0;
};

using StreamingCallback = std::function<void(const StreamingSegment &)>;

class StreamSession {
public:
    struct Impl;
    explicit StreamSession(std::unique_ptr<Impl> impl);
    ~StreamSession();

    StreamSession(const StreamSession &)            = delete;
    StreamSession & operator=(const StreamSession &) = delete;
    StreamSession(StreamSession &&) noexcept;
    StreamSession & operator=(StreamSession &&) noexcept;

    void feed_pcm_f32(const float * samples, int n_samples);
    void feed_pcm_i16(const int16_t * samples, int n_samples);
    void finalize();
    void cancel();

    const StreamingOptions & options() const;

private:
    std::unique_ptr<Impl> pimpl_;
};

// Live speaker-diarization session. Phase 11.11.1 implementation:
// sliding history window + per-chunk diarize(). Each feed_pcm_*()
// pushes audio into a ring; whenever chunk_ms of new audio has arrived,
// the engine runs Sortformer over the last `history_ms` of audio,
// emits segments that overlap the new chunk's time range, and slides
// the chunk pointer forward.
//
// Limitation (Phase 11.11.1): speaker IDs are derived from each
// per-chunk diarize() call independently and may shift slowly across
// chunks until the history window is full. With history_ms >> chunk_ms
// the IDs stabilise quickly. Phase 11.11.2 will add proper NeMo-style
// spkcache compression for fully stable cross-chunk speaker identity.
class SortformerStreamSession {
public:
    struct Impl;
    explicit SortformerStreamSession(std::unique_ptr<Impl> impl);
    ~SortformerStreamSession();

    SortformerStreamSession(const SortformerStreamSession &)            = delete;
    SortformerStreamSession & operator=(const SortformerStreamSession &) = delete;
    SortformerStreamSession(SortformerStreamSession &&) noexcept;
    SortformerStreamSession & operator=(SortformerStreamSession &&) noexcept;

    void feed_pcm_f32(const float * samples, int n_samples);
    void feed_pcm_i16(const int16_t * samples, int n_samples);
    void finalize();
    void cancel();

    const SortformerStreamingOptions & options() const;

private:
    std::unique_ptr<Impl> pimpl_;
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

    EngineResult transcribe_stream(const std::string & wav_path,
                                   const StreamingOptions & opts,
                                   StreamingCallback on_segment);

    EngineResult transcribe_samples_stream(const float * samples,
                                           int n_samples,
                                           int sample_rate,
                                           const StreamingOptions & opts,
                                           StreamingCallback on_segment);

    std::unique_ptr<StreamSession> stream_start(const StreamingOptions & opts,
                                                StreamingCallback on_segment);

    // Diarization (Sortformer models only). Throws if loaded GGUF
    // is a transcription model.
    DiarizationResult diarize(const std::string & wav_path,
                              const DiarizationOptions & opts = {});

    DiarizationResult diarize_samples(const float * samples,
                                      int n_samples,
                                      int sample_rate,
                                      const DiarizationOptions & opts = {});

    // Live diarization session. Throws if the loaded GGUF is not a
    // Sortformer model. See SortformerStreamSession for the limitation
    // on cross-chunk speaker-ID stability in this Phase 11.11.1
    // implementation.
    std::unique_ptr<SortformerStreamSession> diarize_start(
        const SortformerStreamingOptions & opts,
        SortformerSegmentCallback on_segment);

    void cancel();

    // Used by transcribe_with_speakers to slice audio per Sortformer
    // segment and feed each slice through the ASR engine. Public so
    // downstream callers can inspect the underlying model_type and
    // route accordingly.
    bool is_diarization_model() const;
    bool is_transcription_model() const;

    const EngineOptions & options() const;

    // "ctc" or "tdt", reflecting parakeet.model.type metadata of the loaded GGUF.
    std::string model_type() const;

    struct Impl;

private:
    std::unique_ptr<Impl> pimpl_;
};

// Speaker-attributed transcription: runs Sortformer to get speaker
// segments, then transcribes each slice with an ASR engine (CTC or
// TDT) and emits per-segment {speaker_id, text, start_s, end_s}.
//
// Throws std::runtime_error if sortformer_engine is not a Sortformer
// model or asr_engine is not a transcription model. Both engines must
// be loaded at the same sample rate (typically 16 kHz).
AttributedTranscriptionResult transcribe_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const std::string & wav_path,
    const AttributedTranscriptionOptions & opts = {});

AttributedTranscriptionResult transcribe_samples_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const float * samples,
    int n_samples,
    int sample_rate,
    const AttributedTranscriptionOptions & opts = {});

}
