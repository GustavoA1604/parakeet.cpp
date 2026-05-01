// QVAC-18264 — decoder determinism regression gate.
//
// Asserts that running the same decoder N times against the same encoder
// output produces byte-equal results every time, and that the public
// `Engine::transcribe()` / `Engine::diarize()` paths are deterministic
// across repeated calls on the same Engine instance.
//
// Rationale (why this test is needed BEFORE any optimisation change):
//
//   * The follow-up optimization sweep on TDT/EOU/Sortformer plans to
//     replace per-call `std::vector<float>` scratch allocations with
//     reusable scratch buffers owned by the runtime weights / decode
//     state. That change introduces *cross-call state* that doesn't
//     exist today. Without an explicit determinism gate, a leaked
//     scratch entry (e.g. a residual non-zero from the previous call's
//     softmax denominator) could silently bias the next call's output
//     in a way the existing parity-vs-NeMo gate (`max_abs < 5e-3`)
//     happens to swallow.
//
//   * The existing `test-tdt-encoder-parity` /
//     `test-tdt-decoder-parity` / `test-sortformer-parity` /
//     `test-eou-streaming` harnesses each load the model once and
//     run the decode pipeline once, so they wouldn't catch a
//     "second-call drift" the way a multi-call gate does.
//
// What this asserts:
//
//   1. For each model type (CTC / TDT / EOU / Sortformer):
//        * Engine::transcribe(samples, sr) (or Engine::diarize(samples,
//          sr) for Sortformer) called N=5 times in a row produces
//          byte-equal output every call. For CTC/TDT/EOU we compare
//          token_ids (exact integer match) AND the textual transcript;
//          for Sortformer we compare speaker_probs (bit-equal float
//          buffer) AND the segment list (start/end/speaker_id).
//
//   2. Encoder cache hit ratio: total wall time of run K ≤ total wall
//      time of run 0 × 1.10. This catches the tangentially-related
//      regression where a buffer-reuse change accidentally rebuilds
//      the encoder graph each call (since the encoder is shared across
//      all four decoder paths, but its cache key is independent of
//      which decoder is wired up).
//
// What this does NOT assert:
//
//   * Anything about the *correctness* of the output vs. NeMo —
//     `test-tdt-decoder-parity`, `test-tdt-encoder-parity`,
//     `test-sortformer-parity`, `test-eou-streaming` cover that and
//     are explicitly orthogonal to this gate.
//   * Anything about per-call timing within a single run — that's
//     the job of `test-perf-regression` (which is currently
//     CTC/TDT/EOU only; Sortformer's diarize path is gated separately
//     here purely on determinism, since there's no perf harness that
//     understands `Engine::diarize`).
//
// Usage:
//   test-decoder-determinism --model <gguf> --wav <wav> [--runs N] [--n-gpu-layers N]
//
// Returns 0 on success, non-zero on parity failure or setup error.

#include "qvac-parakeet/ctc/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int  n_runs = 5;
    int  n_gpu_layers = 0;
    int  n_threads = 0;
    bool verbose = false;
    // Cache-hit gate compares median(warm runs 1..N-1) to run 0
    // (cold). Median (not max) is the right statistic — a real
    // cache MISS rebuilds the encoder graph each call, lifting
    // EVERY warm run to ≥ cold; thermal-spike one-offs only push
    // max, not median. With median + 1.10x default, CPU thermal
    // jitter on noisy desktops sits well below the gate (we
    // observed median/cold ratios of 0.5-0.9 across all four
    // model types on a 16T Ryzen). Tighten via --cache-hit-ratio
    // on Adreno + warm GGML_OPENCL_CACHE_DIR.
    double cache_hit_ratio_max = 1.10;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [opts]\n"
        "\n"
        "  --model PATH         CTC / TDT / EOU / Sortformer GGUF\n"
        "  --wav PATH           16 kHz mono wav\n"
        "  --runs N             number of repeated calls (default 5; min 2)\n"
        "  --n-gpu-layers N     pass-through to Engine (default 0)\n"
        "  --threads N          CPU threads (0 = HW concurrency)\n"
        "  --cache-hit-ratio F  fail if median(warm enc_ms) exceeds cold\n"
        "                       enc_ms * F. Default 1.10. A real cache MISS\n"
        "                       (graph rebuilt each call) makes EVERY warm\n"
        "                       run ≥ cold; thermal spikes only push max,\n"
        "                       not median, so the gate is robust. Tighten\n"
        "                       on Adreno + warm GGML_OPENCL_CACHE_DIR.\n"
        "  --verbose            print per-run summary\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) o.wav_path   = argv[++i];
        else if (a == "--runs"         && i + 1 < argc) o.n_runs = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads"      && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--cache-hit-ratio" && i + 1 < argc) o.cache_hit_ratio_max = std::atof(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h")    { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty()) { usage(argv[0]); return 2; }
    if (o.n_runs < 2) {
        std::fprintf(stderr, "--runs must be >= 2 (need at least one repeated call to compare)\n");
        return 2;
    }
    return 0;
}

// Minimal RIFF reader (mono 16-bit PCM, matches the helper in
// test_eou_streaming.cpp). We avoid linking the full helper library
// here because this test is intentionally lean.
bool load_wav_pcm(const std::string & path, std::vector<float> & out, int & sr) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char hdr[44];
    if (std::fread(hdr, 1, 44, f) != 44) { std::fclose(f); return false; }
    if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        std::fclose(f); return false;
    }
    sr = *reinterpret_cast<int32_t *>(hdr + 24);
    const int16_t channels = *reinterpret_cast<int16_t *>(hdr + 22);
    const int16_t bits     = *reinterpret_cast<int16_t *>(hdr + 34);
    if (channels != 1 || bits != 16) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_END);
    const long total = std::ftell(f);
    std::fseek(f, 44, SEEK_SET);
    const size_t n = static_cast<size_t>(total - 44) / sizeof(int16_t);
    std::vector<int16_t> pcm(n);
    if (std::fread(pcm.data(), sizeof(int16_t), n, f) != n) { std::fclose(f); return false; }
    std::fclose(f);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = pcm[i] / 32768.0f;
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// ---------- Determinism for CTC / TDT / EOU (transcribe path) ----------

int run_transcribe_path(const Opts & o,
                        const std::vector<float> & samples, int sr,
                        const std::string & model_type_str) {
    qvac_parakeet::EngineOptions eopts;
    eopts.model_gguf_path = o.model_path;
    eopts.n_threads       = o.n_threads;
    eopts.n_gpu_layers    = o.n_gpu_layers;
    eopts.verbose         = false;
    qvac_parakeet::Engine eng(eopts);
    std::fprintf(stderr,
        "[determinism] transcribe path: model=%s backend=%s threads=%d gpu_layers=%d runs=%d\n",
        model_type_str.c_str(),
        eng.backend_name().c_str(),
        o.n_threads, o.n_gpu_layers, o.n_runs);

    std::vector<std::vector<int32_t>> ids_per_run;
    std::vector<std::string>          text_per_run;
    std::vector<double>               enc_ms_per_run;
    ids_per_run.reserve(o.n_runs);
    text_per_run.reserve(o.n_runs);
    enc_ms_per_run.reserve(o.n_runs);

    for (int k = 0; k < o.n_runs; ++k) {
        qvac_parakeet::EngineResult r =
            eng.transcribe_samples(samples.data(), (int) samples.size(), sr);
        if (o.verbose) {
            std::fprintf(stderr,
                "[determinism]   run %d/%d  enc_ms=%.2f  decode_ms=%.2f  total_ms=%.2f  tokens=%zu  text='%s'\n",
                k + 1, o.n_runs,
                r.encoder_ms, r.decode_ms, r.total_ms,
                r.token_ids.size(),
                r.text.size() > 60 ? (r.text.substr(0, 60) + "…").c_str() : r.text.c_str());
        }
        ids_per_run.push_back(std::move(r.token_ids));
        text_per_run.push_back(std::move(r.text));
        enc_ms_per_run.push_back(r.encoder_ms);
    }

    bool ok = true;
    for (int k = 1; k < o.n_runs; ++k) {
        if (ids_per_run[k] != ids_per_run[0]) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d token_ids differ from run 0 "
                "(run0=%zu tokens, runK=%zu tokens)\n",
                k, ids_per_run[0].size(), ids_per_run[k].size());
            const size_t lim = std::min(ids_per_run[k].size(), ids_per_run[0].size());
            for (size_t i = 0; i < lim; ++i) {
                if (ids_per_run[k][i] != ids_per_run[0][i]) {
                    std::fprintf(stderr,
                        "[determinism]   first divergence at index %zu: run0=%d runK=%d\n",
                        i, ids_per_run[0][i], ids_per_run[k][i]);
                    break;
                }
            }
            ok = false;
        }
        if (text_per_run[k] != text_per_run[0]) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d transcript differs from run 0\n"
                "  run 0: '%s'\n  run %d: '%s'\n",
                k, text_per_run[0].c_str(), k, text_per_run[k].c_str());
            ok = false;
        }
    }

    // Encoder cache-hit gate: median(warm) must not re-pay the cold
    // cost. Run 0 is always cold (graph build); runs 1..N-1 are
    // warm and should be at most cold * cache_hit_ratio_max in
    // their median. Median (not max) is the right statistic — a
    // real cache MISS rebuilds the graph each call so EVERY warm
    // run lands at >= cold; thermal-spike one-offs only push max,
    // not median. Skip gate when any enc_ms came back zero (the
    // engine path doesn't separate encoder timing for some model
    // variants).
    bool have_enc_ms = true;
    for (double t : enc_ms_per_run) if (t <= 0.0) { have_enc_ms = false; break; }
    if (have_enc_ms && o.n_runs >= 2) {
        const double cold = enc_ms_per_run[0];
        std::vector<double> warm(enc_ms_per_run.begin() + 1, enc_ms_per_run.end());
        const double med = median(warm);
        const double max_warm = *std::max_element(warm.begin(), warm.end());
        const double min_warm = *std::min_element(warm.begin(), warm.end());
        if (med > cold * o.cache_hit_ratio_max) {
            std::fprintf(stderr,
                "[determinism] FAIL: median(warm) %.2f > cold run0 %.2f * %.2fx "
                "(cache miss? — graph being rebuilt each call?)\n",
                med, cold, o.cache_hit_ratio_max);
            ok = false;
        }
        std::fprintf(stderr,
            "[determinism] enc_ms run0 (cold) = %.2fms; warms min=%.2f median=%.2f max=%.2f (med/cold=%.2fx, max/cold=%.2fx)\n",
            cold, min_warm, med, max_warm, med / cold, max_warm / cold);
    }

    if (!ok) return 1;
    std::fprintf(stderr,
        "[determinism] PASS  transcribe x %d: %zu tokens / %zu chars / encoder cache hit\n",
        o.n_runs, ids_per_run[0].size(), text_per_run[0].size());
    return 0;
}

// ---------- Determinism for Sortformer (diarize path) ----------

bool sort_seg_eq(const qvac_parakeet::DiarizationSegment & a,
                 const qvac_parakeet::DiarizationSegment & b) {
    return a.speaker_id == b.speaker_id &&
           a.start_s == b.start_s &&
           a.end_s   == b.end_s;
}

int run_diarize_path(const Opts & o,
                     const std::vector<float> & samples, int sr,
                     const std::string & model_type_str) {
    qvac_parakeet::EngineOptions eopts;
    eopts.model_gguf_path = o.model_path;
    eopts.n_threads       = o.n_threads;
    eopts.n_gpu_layers    = o.n_gpu_layers;
    eopts.verbose         = false;
    qvac_parakeet::Engine eng(eopts);
    std::fprintf(stderr,
        "[determinism] diarize path: model=%s backend=%s threads=%d gpu_layers=%d runs=%d\n",
        model_type_str.c_str(),
        eng.backend_name().c_str(),
        o.n_threads, o.n_gpu_layers, o.n_runs);

    std::vector<std::vector<float>>    probs_per_run;
    std::vector<std::vector<qvac_parakeet::DiarizationSegment>> segs_per_run;
    std::vector<double>                enc_ms_per_run;
    probs_per_run.reserve(o.n_runs);
    segs_per_run.reserve(o.n_runs);
    enc_ms_per_run.reserve(o.n_runs);

    for (int k = 0; k < o.n_runs; ++k) {
        qvac_parakeet::DiarizationResult r =
            eng.diarize_samples(samples.data(), (int) samples.size(), sr,
                                qvac_parakeet::DiarizationOptions{});
        if (o.verbose) {
            std::fprintf(stderr,
                "[determinism]   run %d/%d  enc_ms=%.2f  decode_ms=%.2f  total_ms=%.2f  "
                "frames=%d num_spks=%d segs=%zu\n",
                k + 1, o.n_runs,
                r.encoder_ms, r.decode_ms, r.total_ms,
                r.n_frames, r.num_spks, r.segments.size());
        }
        probs_per_run.push_back(std::move(r.speaker_probs));
        segs_per_run.push_back(std::move(r.segments));
        enc_ms_per_run.push_back(r.encoder_ms);
    }

    bool ok = true;
    for (int k = 1; k < o.n_runs; ++k) {
        if (probs_per_run[k].size() != probs_per_run[0].size()) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d speaker_probs size differs (%zu vs run0 %zu)\n",
                k, probs_per_run[k].size(), probs_per_run[0].size());
            ok = false;
            continue;
        }
        if (std::memcmp(probs_per_run[k].data(), probs_per_run[0].data(),
                        probs_per_run[k].size() * sizeof(float)) != 0) {
            // Find + report first diff.
            for (size_t i = 0; i < probs_per_run[k].size(); ++i) {
                if (std::memcmp(&probs_per_run[k][i], &probs_per_run[0][i], sizeof(float)) != 0) {
                    std::fprintf(stderr,
                        "[determinism] FAIL: run %d speaker_probs[%zu]=%.9g run0=%.9g delta=%.3e\n",
                        k, i, (double) probs_per_run[k][i], (double) probs_per_run[0][i],
                        (double) (probs_per_run[k][i] - probs_per_run[0][i]));
                    break;
                }
            }
            ok = false;
        }
        if (segs_per_run[k].size() != segs_per_run[0].size()) {
            std::fprintf(stderr,
                "[determinism] FAIL: run %d produced %zu segments vs run0 %zu\n",
                k, segs_per_run[k].size(), segs_per_run[0].size());
            ok = false;
            continue;
        }
        for (size_t i = 0; i < segs_per_run[k].size(); ++i) {
            if (!sort_seg_eq(segs_per_run[k][i], segs_per_run[0][i])) {
                std::fprintf(stderr,
                    "[determinism] FAIL: run %d segment %zu differs "
                    "(spk=%d/%d  start=%.4f/%.4f  end=%.4f/%.4f)\n",
                    k, i,
                    segs_per_run[k][i].speaker_id, segs_per_run[0][i].speaker_id,
                    segs_per_run[k][i].start_s,    segs_per_run[0][i].start_s,
                    segs_per_run[k][i].end_s,      segs_per_run[0][i].end_s);
                ok = false;
            }
        }
    }

    // See run_transcribe_path comment for rationale on median(warm) vs cold.
    bool have_enc_ms = true;
    for (double t : enc_ms_per_run) if (t <= 0.0) { have_enc_ms = false; break; }
    if (have_enc_ms && o.n_runs >= 2) {
        const double cold = enc_ms_per_run[0];
        std::vector<double> warm(enc_ms_per_run.begin() + 1, enc_ms_per_run.end());
        const double med = median(warm);
        const double max_warm = *std::max_element(warm.begin(), warm.end());
        const double min_warm = *std::min_element(warm.begin(), warm.end());
        if (med > cold * o.cache_hit_ratio_max) {
            std::fprintf(stderr,
                "[determinism] FAIL: median(warm) %.2f > cold run0 %.2f * %.2fx "
                "(cache miss? — graph being rebuilt each call?)\n",
                med, cold, o.cache_hit_ratio_max);
            ok = false;
        }
        std::fprintf(stderr,
            "[determinism] enc_ms run0 (cold) = %.2fms; warms min=%.2f median=%.2f max=%.2f (med/cold=%.2fx, max/cold=%.2fx)\n",
            cold, min_warm, med, max_warm, med / cold, max_warm / cold);
    }

    if (!ok) return 1;
    std::fprintf(stderr,
        "[determinism] PASS  diarize x %d: %zu floats in speaker_probs / %zu segments / encoder cache hit\n",
        o.n_runs,
        probs_per_run[0].size(),
        segs_per_run[0].size());
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    Opts o;
    if (int rc = parse_args(argc, argv, o); rc != 0) return rc;

    std::vector<float> samples;
    int sr = 0;
    if (!load_wav_pcm(o.wav_path, samples, sr)) {
        std::fprintf(stderr, "[determinism] failed to load wav: %s\n", o.wav_path.c_str());
        return 3;
    }

    // Probe the model type via the public Engine API — keeps this
    // test linked through the public library boundary, mirroring how
    // the production qvac-parakeet CLI uses it.
    qvac_parakeet::EngineOptions probe_eopts;
    probe_eopts.model_gguf_path = o.model_path;
    probe_eopts.n_threads       = o.n_threads;
    probe_eopts.n_gpu_layers    = o.n_gpu_layers;
    probe_eopts.verbose         = false;
    qvac_parakeet::Engine probe(probe_eopts);
    const std::string mt = probe.model_type();
    std::fprintf(stderr, "[determinism] model_type=%s\n", mt.c_str());

    if (mt == "sortformer" || probe.is_diarization_model()) {
        return run_diarize_path(o, samples, sr, mt);
    }
    return run_transcribe_path(o, samples, sr, mt);
}
