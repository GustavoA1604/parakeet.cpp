#include "energy_vad.h"

#include <algorithm>
#include <cmath>

namespace qvac_parakeet {

EnergyVad::EnergyVad(int sample_rate, int window_ms, int hangover_ms,
                     float threshold_db) {
    sample_rate_ = std::max(1, sample_rate);
    window_n_    = std::max(1, (sample_rate_ * std::max(1, window_ms)) / 1000);
    hangover_n_  = std::max(0, (sample_rate_ * std::max(0, hangover_ms)) / 1000);
    window_n_    = std::min(window_n_, (int) (sizeof(window_sq_) / sizeof(window_sq_[0])));
    const float thresh_rms = std::pow(10.0f, threshold_db / 20.0f);
    thresh_rms_sq_ = thresh_rms * thresh_rms;
    reset();
}

void EnergyVad::reset() {
    total_samples_seen_ = 0;
    window_sum_sq_      = 0.0;
    window_fill_        = 0;
    window_pos_         = 0;
    hangover_count_     = 0;
    state_              = State::Unknown;
    for (int i = 0; i < window_n_; ++i) window_sq_[i] = 0.0f;
}

EnergyVad::Transition EnergyVad::process(const float * samples, int n_samples,
                                         int64_t start_sample) {
    Transition out;
    if (n_samples <= 0) return out;

    for (int i = 0; i < n_samples; ++i) {
        const float s = samples[i];
        const float sq = s * s;

        if (window_fill_ < window_n_) {
            window_sq_[window_pos_] = sq;
            window_sum_sq_ += sq;
            window_pos_ = (window_pos_ + 1) % window_n_;
            ++window_fill_;
        } else {
            window_sum_sq_ -= window_sq_[window_pos_];
            window_sq_[window_pos_] = sq;
            window_sum_sq_ += sq;
            window_pos_ = (window_pos_ + 1) % window_n_;
        }

        ++total_samples_seen_;

        // Need a full window's worth of samples before we can decide.
        if (window_fill_ < window_n_) continue;

        const double mean_sq = window_sum_sq_ / (double) window_n_;
        const bool   above   = mean_sq > thresh_rms_sq_;

        if (above) {
            hangover_count_ = 0;
            if (state_ != State::Speaking) {
                state_ = State::Speaking;
                out.to_state  = state_;
                out.at_sample = start_sample + i;
                out.rms       = std::sqrt((float) mean_sq);
            }
        } else {
            ++hangover_count_;
            if (state_ != State::Silent && hangover_count_ >= hangover_n_) {
                state_ = State::Silent;
                out.to_state  = state_;
                out.at_sample = start_sample + i;
                out.rms       = std::sqrt((float) mean_sq);
            }
        }
    }
    return out;
}

}
