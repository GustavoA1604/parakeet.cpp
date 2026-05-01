#pragma once

// Streaming transcription primitives shared by every ASR engine
// (CTC, TDT, EOU). The Engine that produces these sessions lives in
// <parakeet/engine.h>.
//
// Two flavours of streaming are exposed:
//
//   - Mode 2 (`Engine::transcribe_stream`): caller hands in the full
//     audio up front; the engine runs the offline encoder once, then
//     walks the encoder frames in `chunk_ms` windows and fires
//     `StreamingCallback` per emitted segment. Zero accuracy delta vs
//     the one-shot `transcribe()` path on CTC GGUFs (byte-equal
//     transcripts), bounded WER on TDT, byte-equal on EOU.
//
//   - Mode 3 (`Engine::stream_start` -> `StreamSession`): true duplex
//     push API. Caller pushes PCM over time via `feed_pcm_*()`; each
//     chunk runs cache-aware inference over [left_context + chunk +
//     right_lookahead]. No new GGUF needed -- the offline-trained
//     checkpoint is reused.
//
// The cross-engine `StreamEvent` callback is independent of the
// per-segment callback: consumers can opt into VadStateChanged /
// EndOfTurn signals on top of (or instead of) per-segment text. The
// shape is intentionally engine-agnostic so the same handler works
// across CTC + TDT + EOU + Sortformer.

#include "export.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

// =============================================================================
//  Phase 13 -- cross-engine VAD / EndOfTurn events (StreamEvent)
//
//  Streaming sessions (`StreamSession` for ASR Mode 3, `SortformerStreamSession`
//  for live diarization) can optionally fire small-shape `StreamEvent` calls
//  in addition to the per-chunk segment callbacks they already emit. Sources,
//  gated on the engine type loaded:
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
    // this flag stays false; Phase 13 maps this onto the cross-engine
    // `StreamEventType::EndOfTurn` event with `eot_confidence` filled in.
    bool   is_eou_boundary = false;
    float  eot_confidence  = 0.0f;

    double encoder_ms = 0.0;
    double decode_ms  = 0.0;
};

using StreamingCallback = std::function<void(const StreamingSegment &)>;

class PARAKEET_API StreamSession {
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

}
