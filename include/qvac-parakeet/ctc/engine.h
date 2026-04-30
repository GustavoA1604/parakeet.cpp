#pragma once

// Persistent Parakeet engine -- CTC, TDT, EOU and Sortformer behind
// one umbrella.
//
// Loads the GGUF once and keeps the preprocessor filterbank + encoder
// weights + decoder (CTC head, TDT/EOU prediction+joint, or Sortformer
// transformer+head) + tokenizer (when applicable) + backend resident so
// subsequent calls pay only the mel extraction + encoder + decode cost.
// The header path under <qvac-parakeet/ctc/...> is historical -- the
// `Engine` class auto-detects the model type at load time and dispatches.
//
// Transcription entry points (CTC, TDT, EOU GGUFs):
//
//   1. transcribe()                - full audio in, full text out (one-shot).
//   2. transcribe_stream()         - Mode 2: full audio in up front, segments
//                                    streamed out via callback as they're
//                                    produced. Zero accuracy delta vs
//                                    transcribe().
//   3. stream_start() -> StreamSession
//                                  - Mode 3: true duplex push API. Caller
//                                    pushes PCM over time via feed_pcm_*;
//                                    each chunk runs cache-aware inference
//                                    over [left_context + chunk +
//                                    right_lookahead] using the existing
//                                    offline-trained GGUF (Phase 8). No
//                                    new model checkpoint is required.
//
// Diarization entry points (Sortformer GGUFs):
//
//   4. diarize()                   - full audio in, list of {speaker, start,
//                                    end} segments out.
//   5. diarize_start() -> SortformerStreamSession
//                                  - Phase 11.11.1 sliding-history live
//                                    diarization push API.
//
// Combined ASR + diarization is exposed as a free function:
//   `transcribe_with_speakers(sortformer_engine, asr_engine, ...)`.
//
// Usage (transcription):
//
//     using qvac_parakeet::ctc::Engine;
//     using qvac_parakeet::ctc::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_gguf_path = "models/parakeet-tdt-0.6b-v3.q8_0.gguf";
//     opts.n_threads       = 8;
//
//     Engine engine(opts);
//     for (const auto & wav_path : wavs) {
//         auto result = engine.transcribe(wav_path);
//         std::puts(result.text.c_str());
//     }
//
// Threading model:
//
//   - Concurrent `transcribe()` / `diarize()` / `transcribe_stream()`
//     calls on the same instance are not supported (the encoder's graph
//     allocator is shared mutable state). Wrap an Engine in your own
//     mutex if you need that, or hold one Engine per worker.
//
//   - `cancel()` is safe to call from any thread while another thread
//     is inside `transcribe*` / `diarize*`. It causes the running call
//     to bail out at the next chunk boundary and return.
//
//   - Each new call to `transcribe*` / `diarize*` resets the cancel
//     flag at entry, so a `cancel()` racing with a subsequent
//     `transcribe()` from the *same* thread will be lost. If you need
//     to hard-stop and not start a new call, gate the next entry on
//     your own application-level flag.
//
//   - `~Engine()` does NOT wait for in-flight calls; destroying an
//     Engine while another thread is inside a `transcribe*` call is
//     undefined behaviour. Call `cancel()` and join the working thread
//     before destruction.
//
//   - `~StreamSession()` and `~SortformerStreamSession()` cancel the
//     session; they do NOT call `finalize()`. If you let a session
//     destruct without an explicit `finalize()` call, any audio that
//     hadn't yet rolled into a chunk is dropped, the synthetic
//     `is_final=true` terminator is not emitted (Sortformer), and the
//     final partial-chunk tail segment is not emitted (CTC/TDT
//     Mode 3). Always call `finalize()` if you care about those.
//
// Implementation in src/parakeet_engine.cpp.

#include "../api.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qvac_parakeet {

struct EngineOptions {
    std::string model_gguf_path;

    int n_gpu_layers = 0;
    int n_threads    = 0;

    bool verbose     = false;
};

// Resolved compute device the Engine is actually running on, after the
// load-time backend cascade (CUDA / Metal / Vulkan / OpenCL) and any
// fallbacks (Adreno-tier policy, OpenCL extension probe, missing GPU
// build, kernel-init failure). This is the *post-fallback* truth and
// will not match the user's `EngineOptions::n_gpu_layers` request when
// a fallback occurred.
enum class BackendDevice : int {
    CPU = 0,
    GPU = 1,
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

// =============================================================================
//  Phase 13 -- cross-engine VAD / EndOfTurn events (StreamEvent)
//
//  Streaming sessions (`StreamSession` for ASR Mode 3, `SortformerStreamSession`
//  for live diarization) can optionally fire small-shape `StreamEvent` calls
//  in addition to the per-chunk `StreamingSegment` / `StreamingDiarizationSegment`
//  callbacks they already emit. Sources, gated on the engine type loaded:
//
//    - **EOU** ASR (`parakeet_realtime_eou_120m-v1`): native `<EOU>` token
//      fires `StreamEventType::EndOfTurn` with `eot_confidence = 1.0`.
//    - **Sortformer** diarization: per-chunk threshold-crossings of
//      `max(speaker_probs)` fire `StreamEventType::VadStateChanged`
//      (Speaking / Silent), with `speaker_id = argmax` when entering Speaking.
//    - **CTC / TDT** ASR (no native VAD source in the model): opt-in
//      energy-VAD fallback when `StreamingOptions::enable_energy_vad = true`.
//
//  Both `on_segment` (existing) and `on_event` (new) coexist; consumers that
//  ignore events keep the same behaviour as before. The event API is
//  intentionally shaped to mirror what whisper.cpp's eventual streaming API
//  will emit so consumers can write engine-agnostic event handling.
//
enum class VadState : int {
    Unknown  = 0,
    Speaking = 1,
    Silent   = 2,
};

enum class StreamEventType : int {
    VadStateChanged = 1,
    EndOfTurn       = 2,
};

struct StreamEvent {
    StreamEventType type  = StreamEventType::VadStateChanged;

    // Wall-clock seconds since the streaming session started feeding samples.
    // For chunk-aligned events this is the chunk's emit-end time; for
    // sample-level events (energy-VAD transitions) it is the transition
    // sample boundary in seconds.
    double timestamp_s = 0.0;

    // Index of the chunk that produced the event, when known. -1 when the
    // event was synthesised between chunks (e.g. an energy-VAD silence
    // transition during long quiet inputs).
    int    chunk_index = -1;

    // VadStateChanged fields
    VadState vad_state  = VadState::Unknown;
    int      speaker_id = -1;     // argmax speaker on entering Speaking; -1 otherwise
    float    vad_score  = 0.0f;   // 0..1; provenance-specific (max speaker prob, RMS, ...)

    // EndOfTurn fields
    float    eot_confidence    = 0.0f;  // 0..1; for EOU = 1.0 when `<EOU>` fired
};

using StreamEventCallback = std::function<void(const StreamEvent &)>;

struct StreamingOptions {
    int sample_rate  = 16000;
    int chunk_ms     = 1000;

    int left_context_ms    = 10000;
    int right_lookahead_ms = 2000;

    bool emit_partials = false;

    // Phase 13 -- per-event callback (independent of `on_segment`).
    // Defaults to `nullptr` (no events emitted; back-compat with existing
    // consumers).
    StreamEventCallback on_event = nullptr;

    // Energy-VAD fallback. When true, CTC / TDT sessions will compute a
    // simple RMS-thresholded VAD over the input PCM and fire
    // `StreamEventType::VadStateChanged` events on transitions. Always-on
    // for sessions whose underlying engine (EOU, Sortformer) has its own
    // native VAD source -- those engines' events take priority. Default
    // off; opt-in for CTC/TDT consumers that want VadState events.
    bool  enable_energy_vad = false;

    // Energy-VAD knobs (dB-scale; applies only when enable_energy_vad).
    // Defaults are tuned for clean 16 kHz mono speech: speech enters above
    // -35 dBFS RMS over a 30 ms window, falls back to silent after 200 ms
    // of below-threshold audio.
    float energy_vad_threshold_db = -35.0f;
    int   energy_vad_window_ms    = 30;
    int   energy_vad_hangover_ms  = 200;
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

// One speaker span emitted by SortformerStreamSession. `is_final=true`
// flags the LAST callback the session will fire; it is either:
//   - a real segment from the trailing partial chunk (when audio ends
//     mid-chunk), or
//   - a synthetic terminator with `speaker_id = -1` and `start_s ==
//     end_s` (when audio ended exactly on a chunk boundary, so all
//     real segments were already delivered as `is_final=false`).
// Consumers should treat `speaker_id < 0` as "session done, no new
// segment" and skip any text/append logic for it.
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

    // Phase 13 -- per-event callback (independent of `on_segment`).
    // Sortformer fires `StreamEventType::VadStateChanged` events at
    // chunk granularity using `max(speaker_probs) > threshold` as the
    // VAD signal, and reports the dominant speaker_id on entering
    // Speaking. Defaults to `nullptr` (back-compat).
    StreamEventCallback on_event = nullptr;
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

    // EOU-only: set to true when this segment ends because the decoder
    // emitted the `<EOU>` end-of-utterance token (i.e. the model decided
    // the speaker finished a turn). For CTC / TDT / whisper sessions
    // this flag stays false; Phase 13 will map this onto a cross-engine
    // `OnEndOfTurn` event with `eot_confidence` filled in.
    bool   is_eou_boundary = false;
    float  eot_confidence  = 0.0f;

    double encoder_ms = 0.0;
    double decode_ms  = 0.0;
};

using StreamingCallback = std::function<void(const StreamingSegment &)>;

class QVAC_PARAKEET_API StreamSession {
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
class QVAC_PARAKEET_API SortformerStreamSession {
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

class QVAC_PARAKEET_API Engine {
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

    // "ctc", "tdt", "eou", or "sortformer", reflecting the
    // parakeet.model.type metadata of the loaded GGUF.
    std::string model_type() const;

    // Resolved compute device for this Engine's loaded model. CPU when
    // the build has no GPU backend compiled in, when no GPU was
    // requested (n_gpu_layers <= 0), or when the requested GPU backend
    // refused to initialise (e.g. Adreno-6xx forced to CPU,
    // GGML_OPENCL_ALLOW_UNKNOWN_GPU=1 but the device lacks the
    // required subgroup-size extension). GPU otherwise.
    BackendDevice backend_device() const;

    // Human-readable name of the active backend, e.g. "CUDA0", "Metal",
    // "Vulkan0", "OpenCL", "CPU". Sourced from `ggml_backend_name()`
    // when a GPU backend is active; literal "CPU" otherwise. Stable for
    // the lifetime of the Engine.
    std::string backend_name() const;

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
QVAC_PARAKEET_API AttributedTranscriptionResult transcribe_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const std::string & wav_path,
    const AttributedTranscriptionOptions & opts = {});

QVAC_PARAKEET_API AttributedTranscriptionResult transcribe_samples_with_speakers(
    Engine & sortformer_engine,
    Engine & asr_engine,
    const float * samples,
    int n_samples,
    int sample_rate,
    const AttributedTranscriptionOptions & opts = {});

// Backward-compatibility shim. The library's public namespace was
// `qvac_parakeet::ctc` through v0.1.0-pre; it is now `qvac_parakeet`
// because the same Engine handles CTC, TDT, EOU and Sortformer GGUFs.
// All names in `qvac_parakeet` are visible via the legacy
// `qvac_parakeet::ctc::` qualifier so existing consumer code keeps
// building. New code should use `qvac_parakeet::` directly.
namespace ctc {
    using namespace ::qvac_parakeet;
}

}
