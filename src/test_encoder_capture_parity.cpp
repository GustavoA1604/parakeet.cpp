// Internal regression gate for the round-1 capture_intermediates flag
// (QVAC-17997). The flag was added on `run_encoder()` to let the
// production transcribe / diarize / streaming paths skip 7 host-bound
// `ggml_backend_tensor_get` round-trips per inference. The shipped
// behaviour relies on the invariant:
//
//   For any (model, mel) combination, run_encoder(..., capture=true)
//   and run_encoder(..., capture=false) must produce IDENTICAL
//   `encoder_out` and `logits` outputs. Only the per-stage capture
//   tensors (subsampling_out, block_0_post_*, block_0_out,
//   block_last_out) differ -- they are populated when capture=true
//   and left empty when capture=false.
//
// This test exercises that invariant directly: same model, same mel,
// two `run_encoder` calls with the flag flipped, asserted bit-equal
// on the two output tensors plus the contract on the capture vectors.
//
// Without this gate, a future refactor that subtly changes the
// capture-skip path (e.g. reordering tensor-get calls, accidentally
// stripping `ggml_set_output` markings on the wrong nodes) could
// produce slightly different `encoder_out` bytes that the existing
// test-streaming gate happens to argmax-collapse into the same
// transcript -- masking a real numerical drift.
//
// Built as `test-encoder-capture-parity` via the
// QVAC_PARAKEET_BUILD_TESTS block. Takes the same `--model` /
// `--wav` args as test-encoder.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav>\n"
        "\n"
        "Asserts that run_encoder(model, mel, ..., capture_intermediates=true)\n"
        "and run_encoder(model, mel, ..., capture_intermediates=false) produce\n"
        "bit-equal `encoder_out` and `logits` (CTC GGUFs only). Capture-only\n"
        "vectors are also checked: empty when capture=false, populated when\n"
        "capture=true.\n",
        argv0);
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void report_first_diff(const std::vector<float> & a, const std::vector<float> & b,
                       const char * label) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            std::fprintf(stderr,
                "[%s] FAIL at index %zu: capture=true %.9g  capture=false %.9g  "
                "delta=%.3e\n",
                label, i, (double) a[i], (double) b[i],
                (double) (a[i] - b[i]));
            return;
        }
    }
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] FAIL: size mismatch (%zu vs %zu)\n",
                     label, a.size(), b.size());
    }
}

}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string wav_path;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--wav"   && i + 1 < argc) wav_path   = argv[++i];
        else if (a == "-h" || a == "--help")     { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty() || wav_path.empty()) { usage(argv[0]); return 2; }

    using namespace qvac_parakeet;

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(model_path, model, /*n_threads=*/0,
                                /*n_gpu_layers=*/0, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-encoder-capture-parity] load_from_gguf rc=%d\n", rc);
        return 1;
    }

    if (model.model_type != ParakeetModelType::CTC) {
        std::fprintf(stderr,
            "[test-encoder-capture-parity] skip: only CTC GGUFs carry both "
            "encoder_out and logits in `EncoderOutputs`; got '%s'\n",
            model.model_type == ParakeetModelType::TDT ? "tdt"
            : model.model_type == ParakeetModelType::EOU ? "eou"
            : model.model_type == ParakeetModelType::SORTFORMER ? "sortformer"
            : "unknown");
        return 0;
    }

    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[test-encoder-capture-parity] load_wav rc=%d\n", rc);
        return 1;
    }
    if (sr != model.mel_cfg.sample_rate) {
        std::fprintf(stderr,
            "[test-encoder-capture-parity] FAIL: wav sr %d != model sr %d\n",
            sr, model.mel_cfg.sample_rate);
        return 1;
    }

    std::vector<float> mel;
    int                n_mel_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(),
                                 model.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[test-encoder-capture-parity] compute_log_mel rc=%d\n", rc);
        return 1;
    }

    // Run 1: capture=true (test-encoder default).
    EncoderOutputs out_capture;
    if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                             out_capture, /*max_layers=*/-1,
                             /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[test-encoder-capture-parity] run_encoder(capture=true) rc=%d\n", rc);
        return 1;
    }

    // Run 2: capture=false (production transcribe / streaming default).
    // This call should hit the cached graph from run 1 (same shape,
    // same all_valid). The capture flag is NOT part of the cache key;
    // both runs produce the same compute, only the host-side copy-back
    // differs.
    EncoderOutputs out_nocapture;
    if (int rc = run_encoder(model, mel.data(), n_mel_frames, model.mel_cfg.n_mels,
                             out_nocapture, /*max_layers=*/-1,
                             /*capture_intermediates=*/false); rc != 0) {
        std::fprintf(stderr, "[test-encoder-capture-parity] run_encoder(capture=false) rc=%d\n", rc);
        return 1;
    }

    // Production-relevant tensors must be bit-equal.
    if (out_capture.n_enc_frames != out_nocapture.n_enc_frames ||
        out_capture.d_model      != out_nocapture.d_model      ||
        out_capture.vocab_size   != out_nocapture.vocab_size) {
        std::fprintf(stderr,
            "[test-encoder-capture-parity] FAIL: shape metadata drift "
            "(n_enc=%d/%d  d_model=%d/%d  vocab=%d/%d)\n",
            out_capture.n_enc_frames, out_nocapture.n_enc_frames,
            out_capture.d_model,      out_nocapture.d_model,
            out_capture.vocab_size,   out_nocapture.vocab_size);
        return 1;
    }
    if (!bit_equal(out_capture.encoder_out, out_nocapture.encoder_out)) {
        report_first_diff(out_capture.encoder_out, out_nocapture.encoder_out,
                          "encoder_out");
        return 1;
    }
    if (!bit_equal(out_capture.logits, out_nocapture.logits)) {
        report_first_diff(out_capture.logits, out_nocapture.logits, "logits");
        return 1;
    }

    // Capture-only tensors must be empty when capture=false.
    auto must_be_empty = [&](const std::vector<float> & v, const char * label) {
        if (!v.empty()) {
            std::fprintf(stderr,
                "[test-encoder-capture-parity] FAIL: %s should be empty with "
                "capture=false; got %zu floats\n", label, v.size());
            return false;
        }
        return true;
    };
    if (!must_be_empty(out_nocapture.subsampling_out,    "subsampling_out"))    return 1;
    if (!must_be_empty(out_nocapture.block_0_post_ff1,   "block_0_post_ff1"))   return 1;
    if (!must_be_empty(out_nocapture.block_0_post_attn,  "block_0_post_attn"))  return 1;
    if (!must_be_empty(out_nocapture.block_0_post_conv,  "block_0_post_conv"))  return 1;
    if (!must_be_empty(out_nocapture.block_0_post_ff2,   "block_0_post_ff2"))   return 1;
    if (!must_be_empty(out_nocapture.block_0_out,        "block_0_out"))        return 1;
    if (!must_be_empty(out_nocapture.block_last_out,     "block_last_out"))     return 1;

    // And populated (non-empty + matching shape) when capture=true.
    auto must_be_populated = [&](const std::vector<float> & v, const char * label) {
        if (v.empty()) {
            std::fprintf(stderr,
                "[test-encoder-capture-parity] FAIL: %s should be populated with "
                "capture=true; got empty\n", label);
            return false;
        }
        return true;
    };
    if (!must_be_populated(out_capture.subsampling_out,    "subsampling_out"))    return 1;
    if (!must_be_populated(out_capture.block_0_post_ff1,   "block_0_post_ff1"))   return 1;
    if (!must_be_populated(out_capture.block_0_post_attn,  "block_0_post_attn"))  return 1;
    if (!must_be_populated(out_capture.block_0_post_conv,  "block_0_post_conv"))  return 1;
    if (!must_be_populated(out_capture.block_0_post_ff2,   "block_0_post_ff2"))   return 1;
    if (!must_be_populated(out_capture.block_0_out,        "block_0_out"))        return 1;
    if (!must_be_populated(out_capture.block_last_out,     "block_last_out"))     return 1;

    std::fprintf(stderr,
        "[test-encoder-capture-parity] PASS  encoder_out (%zu floats) and logits "
        "(%zu floats) bit-equal; capture-only tensors empty when capture=false, "
        "populated when capture=true\n",
        out_capture.encoder_out.size(), out_capture.logits.size());
    return 0;
}
