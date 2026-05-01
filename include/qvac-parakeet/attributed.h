#pragma once

// Combined ASR + Sortformer "who said what" attribution.
//
// Runs the Sortformer engine to get speaker segments, then transcribes
// each slice with the ASR engine (CTC / TDT / EOU) and emits per-segment
// `{speaker_id, text, start_s, end_s}`. Two engines (one Sortformer,
// one ASR) feed in; one attributed-segment list comes out.
//
// See the Engine umbrella in <qvac-parakeet/engine.h> for the
// transcription / diarization primitives this layer composes.

#include "export.h"
#include "engine.h"
#include "diarization.h"

#include <string>
#include <vector>

namespace qvac_parakeet {

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

// Speaker-attributed transcription: runs Sortformer to get speaker
// segments, then transcribes each slice with an ASR engine (CTC, TDT
// or EOU) and emits per-segment {speaker_id, text, start_s, end_s}.
//
// Throws std::runtime_error if `sortformer_engine` is not a Sortformer
// model or `asr_engine` is not a transcription model. Both engines must
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

}
