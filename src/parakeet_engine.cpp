#include "qvac-parakeet/ctc/engine.h"

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
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

    TdtRuntimeWeights   tdt_rt;
    bool                tdt_ready = false;

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

    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        if (tdt_prepare_runtime(pimpl_->model, pimpl_->tdt_rt) != 0) {
            throw std::runtime_error("Engine: tdt_prepare_runtime failed");
        }
        pimpl_->tdt_ready = true;
    }
}


Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

const EngineOptions & Engine::options() const {
    return pimpl_->opts;
}

std::string Engine::model_type() const {
    return pimpl_->model.model_type == ParakeetModelType::TDT ? "tdt" : "ctc";
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
    std::vector<int32_t> ids;
    std::string text;
    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        TdtDecodeOptions dopts;
        TdtDecodeResult  dres;
        if (int rc = tdt_greedy_decode(pimpl_->model, pimpl_->tdt_rt,
                                       enc_out.encoder_out.data(),
                                       enc_out.n_enc_frames, enc_out.d_model,
                                       dopts, dres); rc != 0) {
            throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples: tdt_greedy_decode failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        ids  = std::move(dres.token_ids);
        text = std::move(dres.text);
    } else {
        ids  = ctc_greedy_decode(enc_out.logits.data(), enc_out.n_enc_frames,
                                 pimpl_->model.vocab_size, pimpl_->model.blank_id);
        text = detokenize(pimpl_->model.vocab, ids);
    }
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

    const bool is_tdt = (pimpl_->model.model_type == ParakeetModelType::TDT);

    int32_t prev_token = -1;
    TdtDecodeState tdt_state;
    if (is_tdt) tdt_init_state(pimpl_->tdt_rt, (int) pimpl_->model.blank_id, tdt_state);

    int chunk_index = 0;
    bool first_segment = true;

    for (int start = 0; start < T_enc; start += frames_per_window) {
        if (pimpl_->cancel_flag.load()) break;

        int end = start + frames_per_window;
        if (end > T_enc) end = T_enc;

        const auto t_win = clock::now();

        std::vector<int32_t> win_tokens;
        if (is_tdt) {
            TdtDecodeOptions dopts;
            int steps = 0;
            const float * win_enc = enc_out.encoder_out.data()
                                  + static_cast<size_t>(start) * enc_out.d_model;
            if (int rc = tdt_decode_window(pimpl_->model, pimpl_->tdt_rt,
                                           win_enc, end - start, enc_out.d_model,
                                           dopts, tdt_state, win_tokens, steps);
                rc != 0) {
                throw std::runtime_error("qvac_parakeet::ctc::Engine::transcribe_samples_stream: "
                                         "tdt_decode_window failed (rc=" + std::to_string(rc) + ")");
            }
        } else {
            ctc_greedy_decode_window(enc_out.logits.data(),
                                     start, end, vocab, blank,
                                     prev_token, win_tokens, nullptr);
        }

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
    Engine::Impl *      engine_impl = nullptr;
    StreamingOptions    opts;
    StreamingCallback   on_segment;

    int chunk_samples           = 0;
    int left_context_samples    = 0;
    int right_lookahead_samples = 0;

    std::vector<float>   left_history;
    std::vector<float>   pending;

    int     chunk_index    = 0;
    int64_t emitted_samples = 0;
    int32_t prev_token     = -1;
    TdtDecodeState tdt_state;

    std::string             cumulative_text;
    std::vector<int32_t>    cumulative_token_ids;

    bool finalized = false;
    bool cancelled = false;

    void process_window(const float * window_samples, int window_n,
                        int center_start_sample,
                        int center_end_sample,
                        bool is_final_chunk);
    void try_emit_chunks();
    void flush_remainder();
};

void StreamSession::Impl::process_window(const float * window_samples, int window_n,
                                         int center_start_sample,
                                         int center_end_sample,
                                         bool is_final_chunk) {
    if (cancelled) return;
    if (window_n <= 0) return;

    using clock = std::chrono::steady_clock;
    const auto t_chunk = clock::now();

    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_log_mel(window_samples, window_n,
                                 engine_impl->model.mel_cfg,
                                 mel, n_mel_frames); rc != 0) {
        throw std::runtime_error("StreamSession: compute_log_mel failed (rc=" +
                                 std::to_string(rc) + ")");
    }

    EncoderOutputs enc_out;
    if (int rc = run_encoder(engine_impl->model, mel.data(), n_mel_frames,
                             engine_impl->model.mel_cfg.n_mels, enc_out); rc != 0) {
        throw std::runtime_error("StreamSession: run_encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    }

    const double encoder_ms = ms_since(t_chunk);

    const int T_enc = enc_out.n_enc_frames;
    const int sr    = opts.sample_rate;
    const int frame_samples = sr * ENCODER_FRAME_STRIDE_MS / 1000;

    int left_drop_frames     = center_start_sample / frame_samples;
    int center_frame_count   = (center_end_sample - center_start_sample) / frame_samples;
    int right_drop_frames    = T_enc - left_drop_frames - center_frame_count;
    if (is_final_chunk) {
        right_drop_frames = 0;
        center_frame_count = T_enc - left_drop_frames;
    }

    if (left_drop_frames < 0) left_drop_frames = 0;
    if (left_drop_frames > T_enc) left_drop_frames = T_enc;
    if (right_drop_frames < 0) right_drop_frames = 0;
    if (right_drop_frames > T_enc - left_drop_frames) {
        right_drop_frames = T_enc - left_drop_frames;
    }

    const int center_end_frame = T_enc - right_drop_frames;

    const auto t_dec = clock::now();
    std::vector<int32_t> win_tokens;
    if (engine_impl->model.model_type == ParakeetModelType::TDT) {
        TdtDecodeOptions dopts;
        int steps = 0;
        const int n_frames = std::max(0, center_end_frame - left_drop_frames);
        const float * win_enc = enc_out.encoder_out.data()
                              + static_cast<size_t>(left_drop_frames) * enc_out.d_model;
        if (int rc = tdt_decode_window(engine_impl->model, engine_impl->tdt_rt,
                                       win_enc, n_frames, enc_out.d_model,
                                       dopts, tdt_state, win_tokens, steps);
            rc != 0) {
            throw std::runtime_error("StreamSession: tdt_decode_window failed (rc=" +
                                     std::to_string(rc) + ")");
        }
    } else {
        ctc_greedy_decode_window(enc_out.logits.data(),
                                 left_drop_frames, center_end_frame,
                                 engine_impl->model.vocab_size,
                                 engine_impl->model.blank_id,
                                 prev_token, win_tokens, nullptr);
    }

    const size_t prev_cumulative_len = cumulative_text.size();
    cumulative_token_ids.insert(cumulative_token_ids.end(),
                                win_tokens.begin(), win_tokens.end());
    cumulative_text = detokenize(engine_impl->model.vocab, cumulative_token_ids);
    const std::string win_text = cumulative_text.substr(prev_cumulative_len);

    const double decode_ms = ms_since(t_dec);

    if (on_segment) {
        StreamingSegment seg;
        seg.text        = win_text;
        seg.token_ids   = win_tokens;
        seg.start_s     = static_cast<double>(emitted_samples) / sr;
        seg.end_s       = static_cast<double>(emitted_samples +
                                              (center_end_sample - center_start_sample)) / sr;
        seg.chunk_index = chunk_index;
        seg.is_final    = true;
        seg.encoder_ms  = encoder_ms;
        seg.decode_ms   = decode_ms;
        on_segment(seg);
    }

    emitted_samples += (center_end_sample - center_start_sample);
    ++chunk_index;
}

void StreamSession::Impl::try_emit_chunks() {
    if (cancelled) return;
    while (!cancelled &&
           static_cast<int>(pending.size()) >= chunk_samples + right_lookahead_samples) {
        std::vector<float> window;
        window.reserve(left_history.size() + chunk_samples + right_lookahead_samples);
        window.insert(window.end(), left_history.begin(), left_history.end());
        window.insert(window.end(), pending.begin(),
                      pending.begin() + chunk_samples + right_lookahead_samples);

        const int center_start = static_cast<int>(left_history.size());
        const int center_end   = center_start + chunk_samples;

        process_window(window.data(), static_cast<int>(window.size()),
                       center_start, center_end, /*is_final_chunk=*/false);

        left_history.insert(left_history.end(),
                            pending.begin(), pending.begin() + chunk_samples);
        if (static_cast<int>(left_history.size()) > left_context_samples) {
            left_history.erase(left_history.begin(),
                               left_history.end() - left_context_samples);
        }

        pending.erase(pending.begin(), pending.begin() + chunk_samples);
    }
}

void StreamSession::Impl::flush_remainder() {
    if (cancelled) return;
    if (pending.empty()) return;

    std::vector<float> window;
    window.reserve(left_history.size() + pending.size());
    window.insert(window.end(), left_history.begin(), left_history.end());
    window.insert(window.end(), pending.begin(), pending.end());

    const int center_start = static_cast<int>(left_history.size());
    const int center_end   = center_start + static_cast<int>(pending.size());

    process_window(window.data(), static_cast<int>(window.size()),
                   center_start, center_end, /*is_final_chunk=*/true);

    pending.clear();
    left_history.clear();
}

StreamSession::StreamSession(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}

StreamSession::~StreamSession() {
    if (pimpl_ && !pimpl_->finalized && !pimpl_->cancelled) {
        try { pimpl_->cancelled = true; } catch (...) {}
    }
}
StreamSession::StreamSession(StreamSession &&) noexcept = default;
StreamSession & StreamSession::operator=(StreamSession &&) noexcept = default;

const StreamingOptions & StreamSession::options() const {
    return pimpl_->opts;
}

void StreamSession::feed_pcm_f32(const float * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("StreamSession: moved-from session");
    if (pimpl_->finalized) {
        throw std::runtime_error("StreamSession::feed_pcm_f32: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    pimpl_->pending.insert(pimpl_->pending.end(), samples, samples + n_samples);
    pimpl_->try_emit_chunks();
}

void StreamSession::feed_pcm_i16(const int16_t * samples, int n_samples) {
    if (!pimpl_) throw std::runtime_error("StreamSession: moved-from session");
    if (pimpl_->finalized) {
        throw std::runtime_error("StreamSession::feed_pcm_i16: session already finalized");
    }
    if (pimpl_->cancelled) return;
    if (!samples || n_samples <= 0) return;
    const size_t prev = pimpl_->pending.size();
    pimpl_->pending.resize(prev + n_samples);
    constexpr float inv = 1.0f / 32768.0f;
    for (int i = 0; i < n_samples; ++i) {
        pimpl_->pending[prev + i] = static_cast<float>(samples[i]) * inv;
    }
    pimpl_->try_emit_chunks();
}

void StreamSession::finalize() {
    if (!pimpl_) return;
    if (pimpl_->finalized) return;
    pimpl_->finalized = true;
    pimpl_->try_emit_chunks();
    pimpl_->flush_remainder();
}

void StreamSession::cancel() {
    if (!pimpl_) return;
    pimpl_->cancelled = true;
}

std::unique_ptr<StreamSession> Engine::stream_start(const StreamingOptions & opts,
                                                    StreamingCallback on_segment) {
    if (opts.sample_rate != pimpl_->model.mel_cfg.sample_rate) {
        throw std::runtime_error(
            "Engine::stream_start: opts.sample_rate=" + std::to_string(opts.sample_rate) +
            " does not match model rate=" + std::to_string(pimpl_->model.mel_cfg.sample_rate));
    }
    if (opts.chunk_ms <= 0) {
        throw std::runtime_error("Engine::stream_start: chunk_ms must be > 0");
    }
    if (opts.left_context_ms < 0 || opts.right_lookahead_ms < 0) {
        throw std::runtime_error("Engine::stream_start: left_context_ms and right_lookahead_ms must be >= 0");
    }

    auto impl = std::make_unique<StreamSession::Impl>();
    impl->engine_impl  = pimpl_.get();
    impl->opts         = opts;
    impl->on_segment   = std::move(on_segment);

    const int sr = opts.sample_rate;
    impl->chunk_samples           = sr * opts.chunk_ms / 1000;
    impl->left_context_samples    = sr * opts.left_context_ms / 1000;
    impl->right_lookahead_samples = sr * opts.right_lookahead_ms / 1000;

    impl->left_history.reserve(impl->left_context_samples);
    impl->pending.reserve(impl->chunk_samples + impl->right_lookahead_samples);

    if (pimpl_->model.model_type == ParakeetModelType::TDT) {
        tdt_init_state(pimpl_->tdt_rt, (int) pimpl_->model.blank_id, impl->tdt_state);
    }

    return std::make_unique<StreamSession>(std::move(impl));
}

}
