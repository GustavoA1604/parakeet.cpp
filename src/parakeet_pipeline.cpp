#include "qvac-parakeet/ctc/pipeline.h"

#include "parakeet_ctc.h"
#include "parakeet_log.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet {

int transcribe_wav(const TranscribeOptions & opts, TranscribeResult & result) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(opts.model_gguf_path, model, opts.n_threads,
                                opts.n_gpu_layers, opts.verbose); rc != 0) {
        return rc;
    }
    if (model.model_type != ParakeetModelType::CTC) {
        const char * mt = "Sortformer";
        if (model.model_type == ParakeetModelType::TDT) mt = "TDT";
        else if (model.model_type == ParakeetModelType::EOU) mt = "EOU";
        PARAKEET_LOG_ERROR(
            "qvac_parakeet::transcribe_wav: %s is a %s GGUF; this entry point\n"
            "    only handles CTC. Use qvac_parakeet::Engine (see\n"
            "    <qvac-parakeet/ctc/engine.h>) which auto-dispatches to TDT,\n"
            "    EOU, and Sortformer GGUFs.\n",
            opts.model_gguf_path.c_str(), mt);
        return 11;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(opts.wav_path, samples, sr); rc != 0) return rc;
    if (sr != model.mel_cfg.sample_rate) return 10;

    const auto t1 = clock::now();
    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel, n_frames); rc != 0) return rc;
    const double pre_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                             clock::now() - t1).count() / 1000.0;

    const auto t2 = clock::now();
    EncoderOutputs enc_out;
    if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc_out,
                             /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) return rc;
    const double enc_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                             clock::now() - t2).count() / 1000.0;

    const auto t3 = clock::now();
    std::vector<int32_t> ids = ctc_greedy_decode(
        enc_out.logits.data(), enc_out.n_enc_frames, model.vocab_size, model.blank_id);
    result.text = detokenize(model.vocab, ids);
    result.token_ids = std::move(ids);
    const double dec_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                             clock::now() - t3).count() / 1000.0;

    result.preprocess_ms = pre_ms;
    result.encoder_ms    = enc_ms;
    result.decode_ms     = dec_ms;
    result.total_ms      = std::chrono::duration_cast<std::chrono::microseconds>(
                               clock::now() - t0).count() / 1000.0;
    result.audio_samples = (int) samples.size();
    result.sample_rate   = sr;
    result.mel_frames    = n_frames;
    result.encoder_frames = enc_out.n_enc_frames;
    return 0;
}

}
