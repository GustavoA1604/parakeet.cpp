#pragma once

// 16 kHz wav -> 80-channel log-mel spectrogram with per-feature (per-bin)
// CMVN, matching NeMo's AudioToMelSpectrogramPreprocessor.  The mel
// filterbank is loaded from a GGUF tensor (bit-exact with NeMo) rather
// than recomputed in C++, so parity with the PyTorch reference is
// independent of whatever Slaney/HTK rounding the local libm does.
//
// Implementation in src/mel_preprocess.cpp.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet::ctc {

struct MelConfig {
    int sample_rate = 16000;
    int n_fft       = 512;
    int win_length  = 400;
    int hop_length  = 160;
    int n_mels      = 80;

    float preemph              = 0.97f;
    float log_zero_guard_value = 5.960464477539063e-08f;

    std::vector<float> filterbank;
    std::vector<float> window;
};

int load_wav_mono_f32(const std::string & wav_path,
                      std::vector<float>   & out_samples,
                      int                  & out_sample_rate);

int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    std::vector<float> & out_mel,
                    int                & out_n_frames);

void apply_per_feature_cmvn(std::vector<float> & mel,
                            int                  n_frames,
                            int                  n_mels);

}
