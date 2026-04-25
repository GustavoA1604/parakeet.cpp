#include "qvac-parakeet/ctc/engine.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool file_exists(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

bool load_wav_pcm16le_mono(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char riff[4]; f.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) return false;
    f.ignore(4);
    char wave[4]; f.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) return false;

    bool fmt_ok = false; uint16_t channels = 0; uint16_t bits = 0; uint32_t srate = 0;
    std::vector<char> data;
    while (f) {
        char id[4]; f.read(id, 4);
        if (!f) break;
        uint32_t sz = 0; f.read((char *) &sz, 4);
        if (std::memcmp(id, "fmt ", 4) == 0) {
            std::vector<char> hdr(sz);
            f.read(hdr.data(), sz);
            uint16_t fmt = *(uint16_t *) hdr.data();
            channels    = *(uint16_t *) (hdr.data() + 2);
            srate       = *(uint32_t *) (hdr.data() + 4);
            bits        = *(uint16_t *) (hdr.data() + 14);
            if (fmt != 1 || channels != 1 || bits != 16) return false;
            fmt_ok = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            data.resize(sz);
            f.read(data.data(), sz);
            break;
        } else {
            f.ignore(sz);
        }
    }
    if (!fmt_ok || data.empty()) return false;
    sample_rate = (int) srate;
    const int n = (int) (data.size() / 2);
    samples.resize(n);
    const int16_t * s16 = reinterpret_cast<const int16_t *>(data.data());
    for (int i = 0; i < n; ++i) samples[i] = (float) s16[i] / 32768.0f;
    return true;
}

using namespace qvac_parakeet::ctc;

int run_basic(const std::string & gguf_path, const std::string & wav_path) {

    std::vector<float> samples; int sr = 0;
    if (!load_wav_pcm16le_mono(wav_path, samples, sr)) {
        std::fprintf(stderr, "[sf-stream-test] could not load wav %s\n", wav_path.c_str());
        return 1;
    }
    std::fprintf(stderr, "[sf-stream-test] wav=%s samples=%zu sr=%d\n",
                 wav_path.c_str(), samples.size(), sr);

    EngineOptions eopts;
    eopts.model_gguf_path = gguf_path;
    eopts.verbose         = false;
    Engine engine(eopts);
    if (!engine.is_diarization_model()) {
        std::fprintf(stderr, "[sf-stream-test] %s is not a Sortformer model\n", gguf_path.c_str());
        return 2;
    }

    DiarizationResult offline = engine.diarize_samples(
        samples.data(), (int) samples.size(), sr, {});
    std::fprintf(stderr, "[sf-stream-test] offline segments=%zu\n", offline.segments.size());
    for (const auto & s : offline.segments) {
        std::fprintf(stderr, "  offline [%.2f-%.2f] speaker_%d\n",
                     s.start_s, s.end_s, s.speaker_id);
    }

    int n_callbacks = 0;
    int n_finals    = 0;
    double max_end  = 0.0;
    int max_chunk_index = -1;
    std::vector<StreamingDiarizationSegment> all;

    auto on_seg = [&](const StreamingDiarizationSegment & s) {
        ++n_callbacks;
        if (s.is_final) ++n_finals;
        if (s.end_s > max_end) max_end = s.end_s;
        if (s.chunk_index > max_chunk_index) max_chunk_index = s.chunk_index;
        all.push_back(s);
    };

    SortformerStreamingOptions sopts;
    sopts.sample_rate    = sr;
    sopts.chunk_ms       = 2000;
    sopts.history_ms     = 30000;
    sopts.threshold      = 0.5f;
    sopts.min_segment_ms = 200;

    auto session = engine.diarize_start(sopts, on_seg);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> burst_dist(1, 5000);
    size_t off = 0;
    while (off < samples.size()) {
        const int n = std::min<int>(burst_dist(rng), (int) (samples.size() - off));
        session->feed_pcm_f32(samples.data() + off, n);
        off += n;
    }
    session->finalize();

    std::fprintf(stderr,
        "[sf-stream-test] streaming callbacks=%d final_flags=%d max_end=%.3fs chunks=%d\n",
        n_callbacks, n_finals, max_end, max_chunk_index + 1);

    if (n_callbacks == 0) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: no segments emitted\n");
        return 3;
    }
    if (n_finals == 0) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: no is_final segment after finalize()\n");
        return 4;
    }
    const double audio_s = (double) samples.size() / sr;
    if (max_end > audio_s + 0.5) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: max_end=%.3f > audio=%.3f\n",
                     max_end, audio_s);
        return 5;
    }
    if (max_end < audio_s - 5.0) {
        std::fprintf(stderr, "[sf-stream-test] FAIL: max_end=%.3f << audio=%.3f (lost segments?)\n",
                     max_end, audio_s);
        return 6;
    }

    {
        auto session2 = engine.diarize_start(sopts, [](const StreamingDiarizationSegment &) {});
        session2->feed_pcm_f32(samples.data(), (int) std::min<size_t>(samples.size(), 4 * sr));
        session2->cancel();
        session2->cancel();
    }

    std::fprintf(stderr, "[sf-stream-test] PASS\n");
    return 0;
}

}

int main(int argc, char ** argv) {
    std::string gguf = "models/sortformer-4spk-v1.f16.gguf";
    std::string wav  = "test/samples/two-speakers-16k.wav";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--wav" && i + 1 < argc) wav = argv[++i];
    }
    if (!file_exists(gguf)) {
        std::fprintf(stderr, "[sf-stream-test] SKIP: model not found at %s\n", gguf.c_str());
        return 0;
    }
    if (!file_exists(wav)) {
        std::fprintf(stderr, "[sf-stream-test] SKIP: wav not found at %s\n", wav.c_str());
        return 0;
    }
    try {
        return run_basic(gguf, wav);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[sf-stream-test] EXCEPTION: %s\n", e.what());
        return 99;
    }
}
