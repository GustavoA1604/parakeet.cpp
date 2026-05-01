#pragma once

// Speaker diarization primitives (Sortformer GGUFs).
//
// Two flavours, both produced by the `Engine` umbrella in
// <qvac-parakeet/engine.h>:
//
//   - Offline: `Engine::diarize` / `Engine::diarize_samples` -> full audio
//     in, list of `{speaker_id, start_s, end_s}` segments out, plus the
//     raw per-frame `speaker_probs` matrix for callers who want to do
//     their own thresholding.
//
//   - Streaming: `Engine::diarize_start` -> `SortformerStreamSession`
//     (Phase 11.11.1 sliding-history implementation). Each chunk re-runs
//     `Engine::diarize` over the trailing `history_ms` of audio, emits
//     segments overlapping the new chunk, and slides the history pointer
//     forward. See `SortformerStreamSession` for the cross-chunk
//     speaker-ID stability caveat the sliding-history implementation
//     carries.
//
// Sortformer streaming sessions also surface the cross-engine
// `StreamEvent` callback (declared in <qvac-parakeet/streaming.h>) for
// VAD transitions, with the dominant `speaker_id` reported on entering
// Speaking.

#include "export.h"
#include "streaming.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace qvac_parakeet {

struct DiarizationOptions {
    float threshold      = 0.5f;
    int   min_segment_ms = 0;
};

struct DiarizationSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

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

}
