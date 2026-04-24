#include "qvac-parakeet/ctc/engine.h"

#include "parakeet_ctc.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qvac_parakeet::ctc {

namespace {

constexpr int ENCODER_FRAME_STRIDE_MS = 80;

double ms_since(std::chrono::steady_clock::time_point a) {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now() - a).count() / 1000.0;
}

}

struct Engine::Impl {
    EngineOptions       opts;
    ParakeetCtcModel    model;
    std::atomic<bool>   cancel_flag{false};

    Impl() = default;
};

Engine::Engine(const EngineOptions & opts) : pimpl_(std::make_unique<Impl>()) {
    pimpl_->opts = opts;

    const int rc = load_from_gguf(opts.model_gguf_path,
                                  pimpl_->model,
                                  opts.n_threads,
                                  opts.n_gpu_layers,
                                  opts.verbose);
    if (rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine: failed to load GGUF '" +
                                 opts.model_gguf_path +
                                 "' (rc=" + std::to_string(rc) + ")");
    }
}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

void Engine::cancel() {
    pimpl_->cancel_flag.store(true);
}

EngineResult Engine::transcribe(const std::string & wav_path) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe: failed to load wav '" +
                                 wav_path + "' (rc=" + std::to_string(rc) + ")");
    }
    return transcribe_samples(samples.data(), (int) samples.size(), sr);
}

EngineResult Engine::transcribe_samples(const float * samples, int n_samples, int sample_rate) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples: empty input");
    }
    if (sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(pimpl_->model.mel_cfg.sample_rate) + " Hz");
    }

    pimpl_->cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(samples, n_samples, pimpl_->model.mel_cfg,
                                 mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    const auto t_enc = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                             pimpl_->model.mel_cfg.n_mels, enc_out); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double encoder_ms = ms_since(t_enc);

    const auto t_dec = clock::now();
    std::vector<int32_t> ids = ctc_greedy_decode(
        enc_out.logits.data(), enc_out.n_enc_frames,
        pimpl_->model.vocab_size, pimpl_->model.blank_id);
    std::string text = detokenize(pimpl_->model.vocab, ids);
    const double decode_ms = ms_since(t_dec);

    EngineResult result;
    result.text           = std::move(text);
    result.token_ids      = std::move(ids);
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.decode_ms      = decode_ms;
    result.total_ms       = ms_since(t_total);
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.mel_frames     = n_mel_frames;
    result.encoder_frames = enc_out.n_enc_frames;
    return result;
}

EngineResult Engine::transcribe_stream(const std::string & wav_path,
                                       const StreamingOptions & opts,
                                       StreamingCallback on_segment) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_stream: failed to load wav '" +
                                 wav_path + "' (rc=" + std::to_string(rc) + ")");
    }
    return transcribe_samples_stream(samples.data(), (int) samples.size(), sr,
                                     opts, std::move(on_segment));
}

EngineResult Engine::transcribe_samples_stream(const float * samples,
                                               int n_samples,
                                               int sample_rate,
                                               const StreamingOptions & opts,
                                               StreamingCallback on_segment) {
    if (!samples || n_samples <= 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: empty input");
    }
    if (sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: input is " +
                                 std::to_string(sample_rate) + " Hz but model expects " +
                                 std::to_string(pimpl_->model.mel_cfg.sample_rate) + " Hz");
    }
    if (opts.sample_rate != 0 && opts.sample_rate != sample_rate) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: "
                                 "StreamingOptions.sample_rate must match the input sample_rate");
    }
    if (opts.chunk_ms <= 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: "
                                 "StreamingOptions.chunk_ms must be > 0");
    }

    pimpl_->cancel_flag.store(false);

    using clock = std::chrono::steady_clock;
    const auto t_total = clock::now();

    const auto t_mel = clock::now();
    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(samples, n_samples, pimpl_->model.mel_cfg,
                                 mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double preprocess_ms = ms_since(t_mel);

    const auto t_enc = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(pimpl_->model, mel.data(), n_mel_frames,
                             pimpl_->model.mel_cfg.n_mels, enc_out); rc != 0) {
        throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }
    const double encoder_ms = ms_since(t_enc);

    const int T_enc = enc_out.n_enc_frames;
    const int vocab = pimpl_->model.vocab_size;
    const int blank = pimpl_->model.blank_id;

    int frames_per_window = opts.chunk_ms / ENCODER_FRAME_STRIDE_MS;
    if (frames_per_window < 1) frames_per_window = 1;

    EngineResult result;
    result.preprocess_ms  = preprocess_ms;
    result.encoder_ms     = encoder_ms;
    result.audio_samples  = n_samples;
    result.sample_rate    = sample_rate;
    result.mel_frames     = n_mel_frames;
    result.encoder_frames = T_enc;

    const auto t_dec = clock::now();

    int32_t prev_token = -1;
    int chunk_index = 0;
    bool first_segment = true;

    for (int start = 0; start < T_enc; start += frames_per_window) {
        if (pimpl_->cancel_flag.load()) break;

        int end = start + frames_per_window;
        if (end > T_enc) end = T_enc;

        const auto t_win = clock::now();

        std::vector<int32_t> win_tokens;
        ctc_greedy_decode_window(enc_out.logits.data(),
                                 start, end, vocab, blank,
                                 prev_token, win_tokens, nullptr);

        const size_t prev_cumulative_len = result.text.size();
        result.token_ids.insert(result.token_ids.end(),
                                win_tokens.begin(), win_tokens.end());
        result.text = detokenize(pimpl_->model.vocab, result.token_ids);
        const std::string win_text = result.text.substr(prev_cumulative_len);

        const double win_decode_ms = ms_since(t_win);

        if (on_segment) {
            StreamingSegment seg;
            seg.text        = win_text;
            seg.token_ids   = win_tokens;
            seg.start_s     = static_cast<double>(start) * ENCODER_FRAME_STRIDE_MS / 1000.0;
            seg.end_s       = static_cast<double>(end)   * ENCODER_FRAME_STRIDE_MS / 1000.0;
            seg.chunk_index = chunk_index;
            seg.is_final    = true;
            seg.encoder_ms  = first_segment ? encoder_ms : 0.0;
            seg.decode_ms   = win_decode_ms;
            on_segment(seg);
        }

        ++chunk_index;
        first_segment = false;
    }

    result.decode_ms = ms_since(t_dec);
    result.total_ms  = ms_since(t_total);
    return result;
}

struct StreamSession::Impl {
    StreamingOptions  opts;
    StreamingCallback on_segment;
    bool              finalized = false;
    bool              cancelled = false;
};

StreamSession::StreamSession(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}

StreamSession::~StreamSession() = default;
StreamSession::StreamSession(StreamSession &&) noexcept = default;
StreamSession & StreamSession::operator=(StreamSession &&) noexcept = default;

const StreamingOptions & StreamSession::options() const {
    return pimpl_->opts;
}

void StreamSession::feed_pcm_f32(const float *, int) {
    throw std::runtime_error(
        "qvac_parakeet::ctc::StreamSession::feed_pcm_f32: live duplex streaming "
        "requires a cache-aware streaming GGUF; see PROGRESS.md Phase 8.");
}

void StreamSession::feed_pcm_i16(const int16_t *, int) {
    throw std::runtime_error(
        "qvac_parakeet::ctc::StreamSession::feed_pcm_i16: live duplex streaming "
        "requires a cache-aware streaming GGUF; see PROGRESS.md Phase 8.");
}

void StreamSession::finalize() {
    pimpl_->finalized = true;
}

void StreamSession::cancel() {
    pimpl_->cancelled = true;
}

std::unique_ptr<StreamSession> Engine::stream_start(const StreamingOptions & opts,
                                                    StreamingCallback on_segment) {
    if (!pimpl_->model.supports_streaming) {
        throw std::runtime_error(
            "qvac_parakeet::ctc::Engine::stream_start: loaded GGUF does not support "
            "live duplex streaming. This requires a cache-aware streaming checkpoint "
            "with 'parakeet.encoder.streaming.enabled=true' metadata. "
            "See PROGRESS.md Phase 8 for the streaming pipeline milestone. "
            "For full-audio streamed output, use transcribe_stream() instead.");
    }

    auto impl = std::make_unique<StreamSession::Impl>();
    impl->opts       = opts;
    impl->on_segment = std::move(on_segment);
    return std::make_unique<StreamSession>(std::move(impl));
}

}
