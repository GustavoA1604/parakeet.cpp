#include "mel_preprocess.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qvac_parakeet {

int load_wav_mono_f32(const std::string & wav_path,
                      std::vector<float>   & out_samples,
                      int                  & out_sample_rate) {
    drwav wav;
    if (!drwav_init_file(&wav, wav_path.c_str(), nullptr)) {
        std::fprintf(stderr, "error: could not open wav file %s\n", wav_path.c_str());
        return 1;
    }

    const int channels       = static_cast<int>(wav.channels);
    const int sample_rate    = static_cast<int>(wav.sampleRate);
    const drwav_uint64 total = wav.totalPCMFrameCount;

    std::vector<float> interleaved(total * channels);
    const drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, total, interleaved.data());
    drwav_uninit(&wav);

    if (read != total) {
        std::fprintf(stderr, "error: short read from %s\n", wav_path.c_str());
        return 2;
    }

    out_samples.resize(total);
    if (channels == 1) {
        std::memcpy(out_samples.data(), interleaved.data(), total * sizeof(float));
    } else {
        const float inv_c = 1.0f / static_cast<float>(channels);
        for (drwav_uint64 i = 0; i < total; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < channels; ++c) acc += interleaved[i * channels + c];
            out_samples[i] = acc * inv_c;
        }
    }

    out_sample_rate = sample_rate;
    return 0;
}

namespace {

void fft_radix2_inplace(std::complex<float> * data, int n) {
    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;
    if ((1 << log_n) != n) throw std::runtime_error("fft_radix2_inplace: n must be a power of two");

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * 3.14159265358979323846f / len;
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; ++k) {
                const std::complex<float> u = data[i + k];
                const std::complex<float> v = data[i + k + len/2] * w;
                data[i + k]          = u + v;
                data[i + k + len/2]  = u - v;
                w *= wlen;
            }
        }
    }
}

void apply_preemph(std::vector<float> & x, float preemph) {
    if (preemph == 0.0f || x.size() < 2) return;
    for (size_t t = x.size() - 1; t >= 1; --t) {
        x[t] = x[t] - preemph * x[t - 1];
    }
}

std::vector<float> reflect_pad(const std::vector<float> & x, int pad) {
    const int n = static_cast<int>(x.size());
    std::vector<float> out(n + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        const int src = std::min(pad - i, n - 1);
        out[i] = x[src];
    }
    std::memcpy(out.data() + pad, x.data(), n * sizeof(float));
    for (int i = 0; i < pad; ++i) {
        const int src = std::max(n - 2 - i, 0);
        out[pad + n + i] = x[src];
    }
    return out;
}

std::vector<float> make_padded_window(const std::vector<float> & hann400, int n_fft) {
    const int win_length = static_cast<int>(hann400.size());
    const int pad_total  = n_fft - win_length;
    const int pad_left   = pad_total / 2;
    std::vector<float> out(n_fft, 0.0f);
    std::memcpy(out.data() + pad_left, hann400.data(), win_length * sizeof(float));
    return out;
}

}

int compute_log_mel(const float        * samples,
                    int                  n_samples,
                    const MelConfig    & cfg,
                    std::vector<float> & out_mel,
                    int                & out_n_frames) {
    if (n_samples <= 0) return 1;
    if (cfg.filterbank.size() != static_cast<size_t>(cfg.n_mels * (cfg.n_fft / 2 + 1))) {
        std::fprintf(stderr, "mel: unexpected filterbank size (%zu != %d)\n",
                     cfg.filterbank.size(), cfg.n_mels * (cfg.n_fft / 2 + 1));
        return 2;
    }
    if (static_cast<int>(cfg.window.size()) != cfg.win_length) {
        std::fprintf(stderr, "mel: unexpected window size (%zu != %d)\n",
                     cfg.window.size(), cfg.win_length);
        return 3;
    }

    std::vector<float> x(samples, samples + n_samples);
    apply_preemph(x, cfg.preemph);

    const int pad = cfg.n_fft / 2;
    std::vector<float> x_padded = reflect_pad(x, pad);

    const int n_frames = 1 + n_samples / cfg.hop_length;
    out_n_frames = n_frames;

    const int n_bins = cfg.n_fft / 2 + 1;

    std::vector<float> window_padded = make_padded_window(cfg.window, cfg.n_fft);

    std::vector<float> power(n_frames * n_bins);

    std::vector<std::complex<float>> buf(cfg.n_fft);

    for (int t = 0; t < n_frames; ++t) {
        const int start = t * cfg.hop_length;
        for (int i = 0; i < cfg.n_fft; ++i) {
            buf[i] = std::complex<float>(x_padded[start + i] * window_padded[i], 0.0f);
        }
        fft_radix2_inplace(buf.data(), cfg.n_fft);
        for (int k = 0; k < n_bins; ++k) {
            const float re = buf[k].real();
            const float im = buf[k].imag();
            power[t * n_bins + k] = re * re + im * im;
        }
    }

    const int n_mels = cfg.n_mels;
    std::vector<float> mel(n_frames * n_mels);
    const float * fb = cfg.filterbank.data();
    for (int t = 0; t < n_frames; ++t) {
        const float * frame_power = power.data() + t * n_bins;
        float * mel_t = mel.data() + t * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            const float * row = fb + m * n_bins;
            float acc = 0.0f;
            for (int k = 0; k < n_bins; ++k) acc += row[k] * frame_power[k];
            mel_t[m] = acc;
        }
    }

    const float guard = cfg.log_zero_guard_value;
    for (size_t i = 0; i < mel.size(); ++i) {
        mel[i] = std::log(mel[i] + guard);
    }

    const int seq_len = (n_samples + cfg.hop_length - 1) / cfg.hop_length;
    const int valid_frames = std::min(seq_len, n_frames);

    apply_per_feature_cmvn(mel, valid_frames, n_mels);

    for (int t = valid_frames; t < n_frames; ++t) {
        for (int m = 0; m < n_mels; ++m) mel[t * n_mels + m] = 0.0f;
    }

    out_mel = std::move(mel);
    return 0;
}

void apply_per_feature_cmvn(std::vector<float> & mel, int n_valid_frames, int n_mels) {
    if (n_valid_frames <= 0 || n_mels <= 0) return;

    for (int m = 0; m < n_mels; ++m) {
        double sum = 0.0;
        for (int t = 0; t < n_valid_frames; ++t) sum += mel[t * n_mels + m];
        const double mean = sum / n_valid_frames;

        double ss = 0.0;
        for (int t = 0; t < n_valid_frames; ++t) {
            const double d = mel[t * n_mels + m] - mean;
            ss += d * d;
        }
        const double denom = std::max(1, n_valid_frames - 1);
        const double std_ = std::sqrt(ss / denom) + 1e-5;
        const float inv_std = 1.0f / static_cast<float>(std_);
        const float fmean   = static_cast<float>(mean);

        for (int t = 0; t < n_valid_frames; ++t) {
            mel[t * n_mels + m] = (mel[t * n_mels + m] - fmean) * inv_std;
        }
    }
}

}
