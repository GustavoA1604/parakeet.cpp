#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "qvac-parakeet/ctc/engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::mutex              g_mu;
std::condition_variable g_cv;
std::vector<float>      g_pending;
std::atomic<bool>       g_stop{false};

void on_sigint(int) {
    g_stop.store(true);
    g_cv.notify_all();
}

void data_callback(ma_device * /*device*/, void * /*output*/, const void * input, ma_uint32 frame_count) {
    const float * in = static_cast<const float *>(input);
    if (!in || frame_count == 0) return;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        const size_t prev = g_pending.size();
        g_pending.resize(prev + frame_count);
        std::memcpy(g_pending.data() + prev, in, frame_count * sizeof(float));
    }
    g_cv.notify_one();
}

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet-ctc.gguf> [options]\n"
        "\n"
        "Captures the default input device at 16 kHz mono and transcribes live.\n"
        "Press Ctrl-C to stop; the final partial chunk is flushed before exit.\n"
        "\n"
        "options:\n"
        "  --model PATH                   path to the parakeet-ctc GGUF (required)\n"
        "  --n-gpu-layers N               GPU offload (build with -DGGML_METAL=ON etc.)\n"
        "  --threads N                    CPU threads (0 = hardware_concurrency)\n"
        "  --chunk-ms N                   segment stride in ms (default 1000)\n"
        "  --left-context-ms N            left context per chunk in ms (default 5000)\n"
        "  --right-lookahead-ms N         right lookahead per chunk in ms (default 1000)\n"
        "  --list-devices                 list available capture devices and exit\n"
        "  --device N                     use device with this index (default: system default)\n"
        "  --help                         print this help\n",
        argv0);
}

struct Args {
    std::string model_path;
    int  n_gpu_layers = 0;
    int  n_threads    = 0;
    int  chunk_ms     = 1000;
    int  left_ms      = 5000;
    int  right_ms     = 1000;
    bool list_devices = false;
    int  device_index = -1;
};

bool parse_args(int argc, char ** argv, Args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--help" || s == "-h") { print_usage(argv[0]); std::exit(0); }
        else if (s == "--model"              && i + 1 < argc) a.model_path    = argv[++i];
        else if (s == "--n-gpu-layers"       && i + 1 < argc) a.n_gpu_layers  = std::atoi(argv[++i]);
        else if (s == "--threads"            && i + 1 < argc) a.n_threads     = std::atoi(argv[++i]);
        else if (s == "--chunk-ms"           && i + 1 < argc) a.chunk_ms      = std::atoi(argv[++i]);
        else if (s == "--left-context-ms"    && i + 1 < argc) a.left_ms       = std::atoi(argv[++i]);
        else if (s == "--right-lookahead-ms" && i + 1 < argc) a.right_ms      = std::atoi(argv[++i]);
        else if (s == "--list-devices")                       a.list_devices  = true;
        else if (s == "--device"             && i + 1 < argc) a.device_index  = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "unknown option: %s\n", s.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (!a.list_devices && a.model_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

int list_devices_and_exit() {
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_init failed\n");
        return 1;
    }
    ma_device_info * playback_infos = nullptr;
    ma_uint32 n_playback = 0;
    ma_device_info * capture_infos = nullptr;
    ma_uint32 n_capture = 0;
    if (ma_context_get_devices(&ctx, &playback_infos, &n_playback,
                               &capture_infos, &n_capture) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_get_devices failed\n");
        ma_context_uninit(&ctx);
        return 1;
    }
    std::fprintf(stderr, "capture devices:\n");
    for (ma_uint32 i = 0; i < n_capture; ++i) {
        std::fprintf(stderr, "  [%u] %s%s\n", i, capture_infos[i].name,
                     capture_infos[i].isDefault ? " (default)" : "");
    }
    ma_context_uninit(&ctx);
    return 0;
}

}

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;
    if (args.list_devices) return list_devices_and_exit();

    std::fprintf(stderr, "[live-mic] loading %s\n", args.model_path.c_str());
    qvac_parakeet::ctc::EngineOptions eopts;
    eopts.model_gguf_path = args.model_path;
    eopts.n_gpu_layers    = args.n_gpu_layers;
    eopts.n_threads       = args.n_threads;

    qvac_parakeet::ctc::Engine engine(eopts);

    qvac_parakeet::ctc::StreamingOptions sopts;
    sopts.sample_rate        = 16000;
    sopts.chunk_ms           = args.chunk_ms;
    sopts.left_context_ms    = args.left_ms;
    sopts.right_lookahead_ms = args.right_ms;

    auto sess = engine.stream_start(sopts,
        [&](const qvac_parakeet::ctc::StreamingSegment & seg) {
            if (seg.text.empty()) return;
            std::printf("\033[2K\r[%.2f-%.2f]%s\n", seg.start_s, seg.end_s, seg.text.c_str());
            std::fflush(stdout);
        });

    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_context_init failed\n");
        return 1;
    }

    ma_device_info capture_info;
    ma_device_id   chosen_id;
    ma_device_id * chosen_ptr = nullptr;
    if (args.device_index >= 0) {
        ma_device_info * playback_infos = nullptr;
        ma_uint32 n_playback = 0;
        ma_device_info * capture_infos = nullptr;
        ma_uint32 n_capture = 0;
        ma_context_get_devices(&ctx, &playback_infos, &n_playback,
                               &capture_infos, &n_capture);
        if (args.device_index >= (int) n_capture) {
            std::fprintf(stderr, "device index %d out of range (have %u capture devices)\n",
                         args.device_index, n_capture);
            ma_context_uninit(&ctx);
            return 1;
        }
        capture_info = capture_infos[args.device_index];
        chosen_id    = capture_info.id;
        chosen_ptr   = &chosen_id;
        std::fprintf(stderr, "[live-mic] capture device: [%d] %s\n",
                     args.device_index, capture_info.name);
    } else {
        std::fprintf(stderr, "[live-mic] capture device: <system default>\n");
    }

    ma_device_config dcfg      = ma_device_config_init(ma_device_type_capture);
    dcfg.capture.pDeviceID     = chosen_ptr;
    dcfg.capture.format        = ma_format_f32;
    dcfg.capture.channels      = 1;
    dcfg.sampleRate            = 16000;
    dcfg.dataCallback          = data_callback;

    ma_device device;
    if (ma_device_init(&ctx, &dcfg, &device) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_init failed\n");
        ma_context_uninit(&ctx);
        return 1;
    }
    if (ma_device_start(&device) != MA_SUCCESS) {
        std::fprintf(stderr, "ma_device_start failed\n");
        ma_device_uninit(&device);
        ma_context_uninit(&ctx);
        return 1;
    }

    std::signal(SIGINT,  on_sigint);
    std::signal(SIGTERM, on_sigint);

    std::fprintf(stderr,
        "[live-mic] listening at 16 kHz mono.  "
        "chunk=%d ms  left=%d ms  right=%d ms. Speak, Ctrl-C to stop.\n\n",
        args.chunk_ms, args.left_ms, args.right_ms);

    while (!g_stop.load()) {
        std::vector<float> batch;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait_for(lk, std::chrono::milliseconds(100),
                          [] { return !g_pending.empty() || g_stop.load(); });
            if (g_pending.empty()) continue;
            batch.swap(g_pending);
        }
        sess->feed_pcm_f32(batch.data(), (int) batch.size());
    }

    std::fprintf(stderr, "\n[live-mic] stopping...\n");
    ma_device_stop(&device);

    {
        std::vector<float> tail;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            tail.swap(g_pending);
        }
        if (!tail.empty()) sess->feed_pcm_f32(tail.data(), (int) tail.size());
    }

    sess->finalize();

    ma_device_uninit(&device);
    ma_context_uninit(&ctx);
    return 0;
}
