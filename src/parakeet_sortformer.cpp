#include "parakeet_sortformer.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace qvac_parakeet {

namespace {

void dequant(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("sortformer_prepare_runtime: missing tensor");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float)
        throw std::runtime_error(std::string("sortformer_prepare_runtime: no to_float for type ") + ggml_type_name(t->type));
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> host_raw(nbytes);
    ggml_backend_tensor_get(t, host_raw.data(), 0, nbytes);
    tr->to_float(host_raw.data(), out.data(), (int64_t) n);
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// See `parakeet_tdt.cpp::gemv_f32` for the rationale on why
// vectorising this matters even though Sortformer's transformer is
// `tf_d_model = 192` (much smaller than the CTC encoder). The inner
// dot-product runs T*T*n_heads*head_dim FMAs per `self_attention`
// call -- already small but on the critical path of every diarize
// chunk; vectorisation is free with __restrict + `-O3 -ffast-math`
// on gcc/clang and the `#pragma GCC ivdep` removes the data-
// dependence assumption that otherwise pessimises some compilers.
void gemv(const float * __restrict W, const float * __restrict x,
          const float * __restrict b, float * __restrict y,
          int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; ++i) {
        const float * __restrict row = W + (size_t) i * in_dim;
        float acc = b ? b[i] : 0.0f;
        #pragma GCC ivdep
        for (int j = 0; j < in_dim; ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
}

void linear_batch(const float * __restrict W, const float * __restrict X,
                  const float * __restrict b, float * __restrict Y,
                  int n_rows, int out_dim, int in_dim) {
    for (int t = 0; t < n_rows; ++t) {
        gemv(W, X + (size_t) t * in_dim, b, Y + (size_t) t * out_dim, out_dim, in_dim);
    }
}

void layer_norm_inplace(float * X, int n_rows, int d, const float * gamma, const float * beta, float eps = 1e-5f) {
    for (int t = 0; t < n_rows; ++t) {
        float * row = X + (size_t) t * d;
        double sum = 0.0;
        for (int i = 0; i < d; ++i) sum += row[i];
        const float mean = (float) (sum / d);
        double sq = 0.0;
        for (int i = 0; i < d; ++i) { float c = row[i] - mean; sq += c * c; }
        const float inv = 1.0f / std::sqrt((float) (sq / d) + eps);
        for (int i = 0; i < d; ++i) {
            row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

void self_attention(const SortformerTransformerRuntimeBlock & B,
                    int n_heads, int head_dim, int d_model, int T,
                    const float * __restrict X,
                    float * __restrict Y) {
    std::vector<float> Q((size_t) T * d_model);
    std::vector<float> K((size_t) T * d_model);
    std::vector<float> V((size_t) T * d_model);

    linear_batch(B.attn_q_w.data(), X, B.attn_q_b.data(), Q.data(), T, d_model, d_model);
    linear_batch(B.attn_k_w.data(), X, B.attn_k_b.data(), K.data(), T, d_model, d_model);
    linear_batch(B.attn_v_w.data(), X, B.attn_v_b.data(), V.data(), T, d_model, d_model);

    const float scale = 1.0f / std::sqrt((float) head_dim);

    std::vector<float> ctx((size_t) T * d_model, 0.0f);
    std::vector<float> scores((size_t) T);
    std::vector<float> probs((size_t) T);

    for (int h = 0; h < n_heads; ++h) {
        const int off = h * head_dim;
        for (int qi = 0; qi < T; ++qi) {
            const float * q_row = Q.data() + (size_t) qi * d_model + off;
            float max_score = -1e30f;
            for (int kj = 0; kj < T; ++kj) {
                const float * k_row = K.data() + (size_t) kj * d_model + off;
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d) s += q_row[d] * k_row[d];
                s *= scale;
                scores[kj] = s;
                if (s > max_score) max_score = s;
            }
            float sum_exp = 0.0f;
            for (int kj = 0; kj < T; ++kj) {
                const float e = std::exp(scores[kj] - max_score);
                probs[kj] = e;
                sum_exp += e;
            }
            const float inv_sum = 1.0f / sum_exp;
            float * c_row = ctx.data() + (size_t) qi * d_model + off;
            for (int kj = 0; kj < T; ++kj) {
                const float p = probs[kj] * inv_sum;
                const float * v_row = V.data() + (size_t) kj * d_model + off;
                for (int d = 0; d < head_dim; ++d) c_row[d] += p * v_row[d];
            }
        }
    }

    linear_batch(B.attn_o_w.data(), ctx.data(), B.attn_o_b.data(), Y, T, d_model, d_model);
}

void transformer_block(const SortformerTransformerRuntimeBlock & B,
                       int n_heads, int head_dim, int d_model, int T,
                       float * X) {
    std::vector<float> attn_out((size_t) T * d_model);
    self_attention(B, n_heads, head_dim, d_model, T, X, attn_out.data());

    for (size_t i = 0; i < (size_t) T * d_model; ++i) attn_out[i] += X[i];
    layer_norm_inplace(attn_out.data(), T, d_model, B.ln1_w.data(), B.ln1_b.data());

    const int d_ff = (int) B.ffn_in_b.size();
    std::vector<float> ffn_hidden((size_t) T * d_ff);
    linear_batch(B.ffn_in_w.data(), attn_out.data(), B.ffn_in_b.data(), ffn_hidden.data(), T, d_ff, d_model);
    for (size_t i = 0; i < ffn_hidden.size(); ++i) ffn_hidden[i] = std::max(0.0f, ffn_hidden[i]);

    std::vector<float> ffn_out((size_t) T * d_model);
    linear_batch(B.ffn_out_w.data(), ffn_hidden.data(), B.ffn_out_b.data(), ffn_out.data(), T, d_model, d_ff);

    for (size_t i = 0; i < (size_t) T * d_model; ++i) ffn_out[i] += attn_out[i];
    layer_norm_inplace(ffn_out.data(), T, d_model, B.ln2_w.data(), B.ln2_b.data());

    std::memcpy(X, ffn_out.data(), (size_t) T * d_model * sizeof(float));
}

}

int sortformer_prepare_runtime(const ParakeetCtcModel & model, SortformerRuntimeWeights & W) {
    W.D_enc       = model.encoder_cfg.sortformer_fc_d_model;
    W.tf_d        = model.encoder_cfg.sortformer_tf_d_model;
    W.tf_inner    = model.encoder_cfg.sortformer_tf_inner_size;
    W.tf_n_heads  = model.encoder_cfg.sortformer_tf_n_heads;
    W.tf_n_layers = model.encoder_cfg.sortformer_tf_n_layers;
    W.num_spks    = model.encoder_cfg.sortformer_num_spks;
    if (W.tf_n_heads <= 0 || W.tf_d % W.tf_n_heads != 0) {
        std::fprintf(stderr, "sortformer_prepare_runtime: tf_d_model %d not divisible by n_heads %d\n",
                     W.tf_d, W.tf_n_heads);
        return 1;
    }
    W.head_dim = W.tf_d / W.tf_n_heads;

    dequant(model.sortformer.encoder_proj_w, W.proj_w);
    dequant(model.sortformer.encoder_proj_b, W.proj_b);

    W.blocks.clear();
    W.blocks.resize(W.tf_n_layers);
    for (int l = 0; l < W.tf_n_layers; ++l) {
        const auto & sb = model.sortformer.transformer[l];
        auto & rb = W.blocks[l];
        dequant(sb.attn_q_w, rb.attn_q_w); dequant(sb.attn_q_b, rb.attn_q_b);
        dequant(sb.attn_k_w, rb.attn_k_w); dequant(sb.attn_k_b, rb.attn_k_b);
        dequant(sb.attn_v_w, rb.attn_v_w); dequant(sb.attn_v_b, rb.attn_v_b);
        dequant(sb.attn_o_w, rb.attn_o_w); dequant(sb.attn_o_b, rb.attn_o_b);
        dequant(sb.ln1_w,    rb.ln1_w);    dequant(sb.ln1_b,    rb.ln1_b);
        dequant(sb.ffn_in_w, rb.ffn_in_w); dequant(sb.ffn_in_b, rb.ffn_in_b);
        dequant(sb.ffn_out_w,rb.ffn_out_w);dequant(sb.ffn_out_b,rb.ffn_out_b);
        dequant(sb.ln2_w,    rb.ln2_w);    dequant(sb.ln2_b,    rb.ln2_b);
    }

    dequant(model.sortformer.head_h2h_w, W.head_h2h_w);
    dequant(model.sortformer.head_h2h_b, W.head_h2h_b);
    dequant(model.sortformer.head_h2s_w, W.head_h2s_w);
    dequant(model.sortformer.head_h2s_b, W.head_h2s_b);

    return 0;
}

int sortformer_diarize(const ParakeetCtcModel & model,
                       const SortformerRuntimeWeights & W,
                       const float * encoder_out,
                       int T_enc, int D_enc,
                       const SortformerDiarizationOptions & opts,
                       SortformerDiarizationResult & out) {
    if (D_enc != W.D_enc) {
        std::fprintf(stderr, "sortformer_diarize: encoder D mismatch %d vs %d\n", D_enc, W.D_enc);
        return 1;
    }
    if (T_enc <= 0) {
        out.n_frames = 0;
        out.num_spks = W.num_spks;
        out.speaker_probs.clear();
        out.segments.clear();
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<float> X((size_t) T_enc * W.tf_d);
    linear_batch(W.proj_w.data(), encoder_out, W.proj_b.data(), X.data(),
                 T_enc, W.tf_d, W.D_enc);

    for (int l = 0; l < W.tf_n_layers; ++l) {
        transformer_block(W.blocks[l], W.tf_n_heads, W.head_dim, W.tf_d, T_enc, X.data());
    }

    for (size_t i = 0; i < X.size(); ++i) X[i] = std::max(0.0f, X[i]);

    std::vector<float> H1((size_t) T_enc * W.tf_d);
    linear_batch(W.head_h2h_w.data(), X.data(), W.head_h2h_b.data(), H1.data(),
                 T_enc, W.tf_d, W.tf_d);

    for (size_t i = 0; i < H1.size(); ++i) H1[i] = std::max(0.0f, H1[i]);

    out.n_frames = T_enc;
    out.num_spks = W.num_spks;
    out.speaker_probs.assign((size_t) T_enc * W.num_spks, 0.0f);
    linear_batch(W.head_h2s_w.data(), H1.data(), W.head_h2s_b.data(), out.speaker_probs.data(),
                 T_enc, W.num_spks, W.tf_d);

    for (auto & v : out.speaker_probs) v = sigmoidf(v);

    out.frame_stride_s = (double) (model.mel_cfg.hop_length *
                                   model.encoder_cfg.subsampling_factor) /
                         (double) model.mel_cfg.sample_rate;

    out.segments.clear();
    const float thr = opts.threshold;
    for (int s = 0; s < W.num_spks; ++s) {
        bool active = false;
        int  start_frame = 0;
        for (int t = 0; t < T_enc; ++t) {
            const bool a = out.speaker_probs[(size_t) t * W.num_spks + s] > thr;
            if (a && !active) { start_frame = t; active = true; }
            if (!a && active) {
                SortformerSegment seg;
                seg.speaker_id = s;
                seg.start_s = start_frame * out.frame_stride_s;
                seg.end_s   = t           * out.frame_stride_s;
                out.segments.push_back(seg);
                active = false;
            }
        }
        if (active) {
            SortformerSegment seg;
            seg.speaker_id = s;
            seg.start_s = start_frame * out.frame_stride_s;
            seg.end_s   = T_enc       * out.frame_stride_s;
            out.segments.push_back(seg);
        }
    }
    std::sort(out.segments.begin(), out.segments.end(),
              [](const SortformerSegment & a, const SortformerSegment & b) {
                  if (a.start_s != b.start_s) return a.start_s < b.start_s;
                  return a.speaker_id < b.speaker_id;
              });

    out.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t0).count() / 1000.0;
    return 0;
}

}
