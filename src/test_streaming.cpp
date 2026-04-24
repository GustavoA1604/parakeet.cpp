#include "qvac-parakeet/ctc/engine.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace qvac_parakeet::ctc;

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int  n_gpu_layers = 0;
    int  n_threads    = 0;
    bool verbose      = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet-ctc.gguf> --wav <input.wav> [options]\n"
        "\n"
        "validates Mode 2 (transcribe_stream) byte-equality vs Mode 1 (transcribe)\n"
        "across a range of chunk sizes, plus the Mode 3 (stream_start) error path.\n"
        "\n"
        "options:\n"
        "  --n-gpu-layers N     offload to GPU backend when > 0\n"
        "  --threads N          CPU threads (0 = hardware_concurrency)\n"
        "  --verbose            print per-test segment counts and timings\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav" && i + 1 < argc) o.wav_path = argv[++i];
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (o.model_path.empty() || o.wav_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

bool approx_equal(double a, double b, double tol_ms) {
    return std::abs(a - b) * 1000.0 <= tol_ms;
}

}

int main(int argc, char ** argv) {
    Opts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;

    EngineOptions eopts;
    eopts.model_gguf_path = opts.model_path;
    eopts.n_gpu_layers    = opts.n_gpu_layers;
    eopts.n_threads       = opts.n_threads;
    eopts.verbose         = opts.verbose;

    std::fprintf(stderr, "[test-streaming] loading %s\n", opts.model_path.c_str());
    Engine engine(eopts);

    std::fprintf(stderr, "[test-streaming] Mode 1 reference: transcribe(%s)\n",
                 opts.wav_path.c_str());
    EngineResult ref = engine.transcribe(opts.wav_path);
    std::fprintf(stderr,
                 "[test-streaming] Mode 1: %d encoder frames, %zu tokens, %.1f ms total\n"
                 "[test-streaming] Mode 1 text: \"%.120s%s\"\n",
                 ref.encoder_frames, ref.token_ids.size(), ref.total_ms,
                 ref.text.c_str(), ref.text.size() > 120 ? "..." : "");

    const int audio_ms = (int) (1000.0 * (double) ref.audio_samples / (double) ref.sample_rate);
    const std::vector<int> chunk_ms_list = {250, 500, 1000, 2000, 4000, audio_ms};

    int failures = 0;
    for (int chunk_ms : chunk_ms_list) {
        if (chunk_ms <= 0) continue;
        StreamingOptions sopts;
        sopts.sample_rate = ref.sample_rate;
        sopts.chunk_ms    = chunk_ms;

        int n_segments = 0;
        double last_end_s = 0.0;
        bool timestamps_ok = true;
        std::string concat_text;

        EngineResult stream_result = engine.transcribe_stream(
            opts.wav_path, sopts,
            [&](const StreamingSegment & seg) {
                if (seg.chunk_index != n_segments) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: chunk_index jump "
                        "(expected %d, got %d)\n",
                        chunk_ms, n_segments, seg.chunk_index);
                    timestamps_ok = false;
                }
                if (!approx_equal(seg.start_s, last_end_s, 1.0)) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d start_s=%.3f "
                        "does not match previous end_s=%.3f\n",
                        chunk_ms, seg.chunk_index, seg.start_s, last_end_s);
                    timestamps_ok = false;
                }
                if (seg.end_s <= seg.start_s) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d has end_s=%.3f <= start_s=%.3f\n",
                        chunk_ms, seg.chunk_index, seg.end_s, seg.start_s);
                    timestamps_ok = false;
                }
                if (!seg.is_final) {
                    std::fprintf(stderr,
                        "[test-streaming] FAIL chunk_ms=%d: seg %d is_final=false "
                        "(Phase 1 Mode 2 must emit final segments only)\n",
                        chunk_ms, seg.chunk_index);
                    timestamps_ok = false;
                }
                concat_text += seg.text;
                last_end_s = seg.end_s;
                ++n_segments;
            });

        const double audio_s = (double) ref.audio_samples / (double) ref.sample_rate;
        if (!approx_equal(last_end_s, audio_s, 100.0)) {
            std::fprintf(stderr,
                "[test-streaming] WARN chunk_ms=%d: last end_s=%.3f differs "
                "from audio duration=%.3f (tolerance 100ms for frame-stride rounding)\n",
                chunk_ms, last_end_s, audio_s);
        }

        if (concat_text != ref.text) {
            std::fprintf(stderr,
                "[test-streaming] FAIL chunk_ms=%d: concatenated segment text "
                "does not match Mode 1 reference\n", chunk_ms);
            std::fprintf(stderr, "  ref:    \"%.200s%s\"\n",
                         ref.text.c_str(), ref.text.size() > 200 ? "..." : "");
            std::fprintf(stderr, "  stream: \"%.200s%s\"\n",
                         concat_text.c_str(), concat_text.size() > 200 ? "..." : "");
            ++failures;
        } else if (!timestamps_ok) {
            ++failures;
        } else {
            std::fprintf(stderr,
                "[test-streaming] PASS chunk_ms=%4d: %3d segments, %.1f ms stream total, "
                "text byte-equal\n",
                chunk_ms, n_segments, stream_result.total_ms);
        }
    }

    bool mode3_ok = false;
    try {
        StreamingOptions sopts;
        sopts.sample_rate = 16000;
        sopts.chunk_ms    = 1000;
        auto sess = engine.stream_start(sopts, [](const StreamingSegment &) {});
        (void)sess;
        std::fprintf(stderr,
            "[test-streaming] FAIL Mode 3: stream_start() unexpectedly succeeded "
            "on a non-streaming GGUF\n");
    } catch (const std::exception & e) {
        const std::string msg = e.what();
        if (msg.find("streaming") != std::string::npos ||
            msg.find("Phase 8")   != std::string::npos) {
            std::fprintf(stderr,
                "[test-streaming] PASS Mode 3: stream_start() errored with "
                "expected message\n");
            mode3_ok = true;
        } else {
            std::fprintf(stderr,
                "[test-streaming] FAIL Mode 3: stream_start() errored with "
                "unexpected message: %s\n", e.what());
        }
    }
    if (!mode3_ok) ++failures;

    if (failures == 0) {
        std::fprintf(stderr, "[test-streaming] all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "[test-streaming] %d check(s) failed\n", failures);
    return 1;
}
