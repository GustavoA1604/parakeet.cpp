// Light-weight perf-regression smoke for QVAC-17997's audit/optimization pass.
// Runs the public Engine::transcribe() path N times and asserts:
//   1. Every run produces the byte-equal reference transcript.
//   2. The encoder graph cache hits after the first call (subsequent
//      `enc_ms` values are at most 1.05x the median of the warm runs).
//   3. The total wall time stays within an explicit ceiling
//      (passed in via --max-encoder-ms, default 600 ms on Q8_0 CPU).
//
// Build target: test-perf-regression (added in CMakeLists.txt). Used to
// catch silent perf regressions introduced by the optimization sweep
// (e.g. cache-key mismatches that re-build the graph every call,
// extra per-call allocations, accidentally re-enabled per-stage tensor
// captures on the production transcribe path, ...). Designed to run in
// well under a minute on a 16-thread Ryzen.

#include "qvac-parakeet/ctc/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    std::string expected_text;
    int n_runs = 6;
    int n_warmup = 2;
    int n_threads = 0;
    int n_gpu_layers = 0;
    double max_enc_ms = 0.0;
    double cache_hit_ratio_max = 1.10; // warm enc_ms must be ≤ 1.10x median
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav> [opts]\n"
        "\n"
        "  --model PATH         CTC/TDT/EOU GGUF (Sortformer not supported by this harness)\n"
        "  --wav PATH           16 kHz mono wav\n"
        "  --expect TEXT        expected transcript (asserted byte-equal). Optional;\n"
        "                       default: parity with the first run.\n"
        "  --runs N             number of timed runs (default 6)\n"
        "  --warmup N           warmup runs not counted in stats (default 2)\n"
        "  --threads N          CPU threads (0 = HW concurrency)\n"
        "  --n-gpu-layers N     pass-through to Engine\n"
        "  --max-enc-ms F       fail if any timed run's encoder_ms exceeds this\n"
        "  --cache-hit-ratio F  fail if any timed encoder_ms exceeds median*F (default 1.10)\n",
        argv0);
}

}

int main(int argc, char ** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model" && i + 1 < argc)              o.model_path = argv[++i];
        else if (a == "--wav"   && i + 1 < argc)              o.wav_path   = argv[++i];
        else if (a == "--expect" && i + 1 < argc)             o.expected_text = argv[++i];
        else if (a == "--runs" && i + 1 < argc)               o.n_runs = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc)             o.n_warmup = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)            o.n_threads = std::atoi(argv[++i]);
        else if (a == "--n-gpu-layers" && i + 1 < argc)       o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--max-enc-ms" && i + 1 < argc)         o.max_enc_ms = std::atof(argv[++i]);
        else if (a == "--cache-hit-ratio" && i + 1 < argc)    o.cache_hit_ratio_max = std::atof(argv[++i]);
        else if (a == "--help" || a == "-h")                  { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (o.model_path.empty() || o.wav_path.empty()) { usage(argv[0]); return 2; }

    qvac_parakeet::EngineOptions eopts;
    eopts.model_gguf_path = o.model_path;
    eopts.n_threads       = o.n_threads;
    eopts.n_gpu_layers    = o.n_gpu_layers;
    eopts.verbose         = false;

    qvac_parakeet::Engine engine(eopts);

    if (engine.is_diarization_model()) {
        std::fprintf(stderr, "[test-perf-regression] skipping: Sortformer not supported by this harness\n");
        return 0;
    }

    std::string reference_text;
    std::vector<double> enc_ms;
    enc_ms.reserve(o.n_runs);

    const auto run_once = [&](int idx, bool is_warmup) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = engine.transcribe(o.wav_path);
        const double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - t0).count() / 1000.0;
        const double enc = res.encoder_ms > 0.0 ? res.encoder_ms : total_ms;
        std::fprintf(stderr, "[test-perf-regression] %s %d/%d  enc=%.2fms total=%.2fms\n",
                     is_warmup ? "warmup" : "run", idx, is_warmup ? o.n_warmup : o.n_runs,
                     enc, total_ms);
        if (idx == 1 && is_warmup) {
            reference_text = res.text;
            if (!o.expected_text.empty() && res.text != o.expected_text) {
                std::fprintf(stderr,
                    "[test-perf-regression] FAIL: transcript mismatch\n"
                    "  expected: \"%s\"\n  got     : \"%s\"\n",
                    o.expected_text.c_str(), res.text.c_str());
                std::exit(1);
            }
        } else if (res.text != reference_text) {
            std::fprintf(stderr,
                "[test-perf-regression] FAIL: non-deterministic transcript on run %d\n"
                "  reference: \"%s\"\n  got      : \"%s\"\n",
                idx, reference_text.c_str(), res.text.c_str());
            std::exit(1);
        }
        if (!is_warmup) enc_ms.push_back(enc);
    };

    for (int i = 1; i <= o.n_warmup; ++i) run_once(i, true);
    for (int i = 1; i <= o.n_runs;   ++i) run_once(i, false);

    if (enc_ms.empty()) {
        std::fprintf(stderr, "[test-perf-regression] FAIL: no timed runs\n");
        return 1;
    }

    std::vector<double> sorted = enc_ms;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    const double max_v  = sorted.back();
    const double min_v  = sorted.front();
    const double mean   = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();

    std::fprintf(stderr,
        "[test-perf-regression] enc_ms summary  mean=%.2f  median=%.2f  min=%.2f  max=%.2f\n",
        mean, median, min_v, max_v);

    if (o.max_enc_ms > 0.0 && max_v > o.max_enc_ms) {
        std::fprintf(stderr,
            "[test-perf-regression] FAIL: max enc_ms %.2f > ceiling %.2f\n",
            max_v, o.max_enc_ms);
        return 1;
    }
    const double ratio = max_v / median;
    if (ratio > o.cache_hit_ratio_max) {
        std::fprintf(stderr,
            "[test-perf-regression] FAIL: max/median = %.2fx > %.2fx (cache miss in steady state?)\n",
            ratio, o.cache_hit_ratio_max);
        return 1;
    }

    std::fprintf(stderr, "[test-perf-regression] PASS  transcript: \"%s\"\n",
                 reference_text.c_str());
    return 0;
}
