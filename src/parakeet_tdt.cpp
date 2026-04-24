#include "parakeet_tdt.h"
#include "sentencepiece_bpe.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace qvac_parakeet::ctc {

namespace {

void dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("tdt_prepare_runtime: missing tensor");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("tdt_prepare_runtime: no to_float for type ") +
                                 ggml_type_name(t->type));
    }
    tr->to_float(t->data, out.data(), (int64_t) n);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void gemv_f32(const float * __restrict W, const float * __restrict x,
              const float * __restrict b, float * __restrict y,
              int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * row = W + (size_t) i * in_dim;
        float acc = b ? b[i] : 0.0f;
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}

void gemv_add_f32(const float * __restrict W, const float * __restrict x,
                  float * __restrict y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * row = W + (size_t) i * in_dim;
        float acc = 0.0f;
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] += acc;
    }
}

void lstm_step(const TdtRuntimeWeights & W,
               const float * __restrict x_input,
               float * __restrict h_state,
               float * __restrict c_state,
               std::vector<float> & scratch) {
    const int H = W.H_pred;
    const int L = W.L;
    const int G = 4 * H;

    scratch.resize((size_t) G);

    const float * x = x_input;
    std::vector<float> layer_input(H);

    for (int layer = 0; layer < L; ++layer) {
        const auto & w = W.lstm[layer];
        const float * h_l = h_state + (size_t) layer * H;
        float * c_l = c_state + (size_t) layer * H;

        gemv_f32(w.w_ih.data(), x, w.b_ih.data(), scratch.data(), G, H);
        for (int i = 0; i < G; ++i) scratch[i] += w.b_hh[i];
        gemv_add_f32(w.w_hh.data(), h_l, scratch.data(), G, H);

        float * h_new = (float *) (h_state + (size_t) layer * H);
        for (int i = 0; i < H; ++i) {
            const float i_g = sigmoid(scratch[0 * H + i]);
            const float f_g = sigmoid(scratch[1 * H + i]);
            const float g_g = std::tanh(scratch[2 * H + i]);
            const float o_g = sigmoid(scratch[3 * H + i]);
            const float c_new = f_g * c_l[i] + i_g * g_g;
            c_l[i] = c_new;
            h_new[i] = o_g * std::tanh(c_new);
        }

        std::memcpy(layer_input.data(), h_new, (size_t) H * sizeof(float));
        x = layer_input.data();
    }
}

void joint_step(const TdtRuntimeWeights & W,
                const float * __restrict enc,
                const float * __restrict pred,
                std::vector<float> & hidden,
                std::vector<float> & logits) {
    const int H   = W.H_joint;
    const int De  = W.D_enc;
    const int Hp  = W.H_pred;
    const int Vo  = W.V_out;

    hidden.resize(H);
    gemv_f32(W.joint_enc_w.data(),  enc,  W.joint_enc_b.data(),  hidden.data(), H, De);
    {
        std::vector<float> tmp(H);
        gemv_f32(W.joint_pred_w.data(), pred, W.joint_pred_b.data(), tmp.data(), H, Hp);
        for (int i = 0; i < H; ++i) hidden[i] += tmp[i];
    }
    for (int i = 0; i < H; ++i) hidden[i] = std::max(0.0f, hidden[i]);

    logits.resize(Vo);
    gemv_f32(W.joint_out_w.data(), hidden.data(), W.joint_out_b.data(), logits.data(), Vo, H);
}

int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

}

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & W) {
    W.H_pred        = model.encoder_cfg.tdt_pred_hidden;
    W.H_joint       = model.encoder_cfg.tdt_joint_hidden;
    W.D_enc         = model.encoder_cfg.d_model;
    W.L             = model.encoder_cfg.tdt_pred_rnn_layers;
    W.num_durations = model.encoder_cfg.tdt_num_durations;
    W.V_plus_1      = (int) model.vocab_size + 1;
    W.V_out         = W.V_plus_1 + W.num_durations;

    dequantize_to_f32(model.tdt.predict_embed, W.embed);

    W.lstm.clear();
    W.lstm.resize(W.L);
    for (int l = 0; l < W.L; ++l) {
        dequantize_to_f32(model.tdt.lstm[l].w_ih, W.lstm[l].w_ih);
        dequantize_to_f32(model.tdt.lstm[l].w_hh, W.lstm[l].w_hh);
        dequantize_to_f32(model.tdt.lstm[l].b_ih, W.lstm[l].b_ih);
        dequantize_to_f32(model.tdt.lstm[l].b_hh, W.lstm[l].b_hh);
    }

    dequantize_to_f32(model.tdt.joint_enc_w,  W.joint_enc_w);
    dequantize_to_f32(model.tdt.joint_enc_b,  W.joint_enc_b);
    dequantize_to_f32(model.tdt.joint_pred_w, W.joint_pred_w);
    dequantize_to_f32(model.tdt.joint_pred_b, W.joint_pred_b);
    dequantize_to_f32(model.tdt.joint_out_w,  W.joint_out_w);
    dequantize_to_f32(model.tdt.joint_out_b,  W.joint_out_b);

    return 0;
}

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      const TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result) {
    if (D_enc != W.D_enc) {
        std::fprintf(stderr, "tdt_greedy_decode: encoder d_model mismatch (%d vs %d)\n",
                     D_enc, W.D_enc);
        return 1;
    }
    const int H     = W.H_pred;
    const int L     = W.L;
    const int V_p1  = W.V_plus_1;
    const int D_n   = W.num_durations;
    const int blank = (int) model.blank_id;

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<float> h_state((size_t) L * H, 0.0f);
    std::vector<float> c_state((size_t) L * H, 0.0f);

    std::vector<float> scratch_lstm;
    std::vector<float> scratch_joint_hidden;
    std::vector<float> scratch_joint_logits;
    std::vector<float> pred_out(H);

    {
        const float * embed_row = W.embed.data() + (size_t) blank * H;
        lstm_step(W, embed_row, h_state.data(), c_state.data(), scratch_lstm);
        std::memcpy(pred_out.data(), h_state.data() + (size_t) (L - 1) * H,
                    (size_t) H * sizeof(float));
    }

    result.token_ids.clear();
    result.token_ids.reserve(T_enc);

    int t = 0;
    int symbols_this_step = 0;
    int total_steps = 0;

    while (t < T_enc) {
        const float * enc_frame = encoder_out + (size_t) t * D_enc;
        joint_step(W, enc_frame, pred_out.data(), scratch_joint_hidden, scratch_joint_logits);
        ++total_steps;

        const int best_token = argmax_f32(scratch_joint_logits.data(), V_p1);
        const int best_dur_idx = argmax_f32(scratch_joint_logits.data() + V_p1, D_n);
        const int best_dur = model.tdt_durations.empty()
                               ? best_dur_idx
                               : model.tdt_durations[best_dur_idx];

        if (best_token == blank) {
            t += std::max(1, best_dur);
            symbols_this_step = 0;
            continue;
        }

        result.token_ids.push_back((int32_t) best_token);

        const float * embed_row = W.embed.data() + (size_t) best_token * H;
        lstm_step(W, embed_row, h_state.data(), c_state.data(), scratch_lstm);
        std::memcpy(pred_out.data(), h_state.data() + (size_t) (L - 1) * H,
                    (size_t) H * sizeof(float));

        ++symbols_this_step;
        if (best_dur > 0 || symbols_this_step >= opts.max_symbols_per_step) {
            t += std::max(1, best_dur);
            symbols_this_step = 0;
        }
    }

    result.text = detokenize(model.vocab, result.token_ids);
    result.steps = total_steps;
    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

}
