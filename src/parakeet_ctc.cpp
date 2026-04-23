#include "parakeet_ctc.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qvac_parakeet::ctc {

struct ParakeetCtcModel::Impl {
    gguf_context     * gguf           = nullptr;
    ggml_context     * ctx            = nullptr;
    ggml_backend_t     backend        = nullptr;
    ggml_gallocr_t     encoder_alloc  = nullptr;
    int                encoder_alloc_T_mel = 0;

    ~Impl() {
        if (encoder_alloc) ggml_gallocr_free(encoder_alloc);
        if (ctx)           ggml_free(ctx);
        if (gguf)          gguf_free(gguf);
        if (backend)       ggml_backend_free(backend);
    }
};

namespace {

int find_key(const gguf_context * g, const std::string & k) {
    return (int) gguf_find_key(g, k.c_str());
}

uint32_t get_u32(const gguf_context * g, const std::string & k, uint32_t fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

float get_f32(const gguf_context * g, const std::string & k, float fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_f32(g, id);
}

bool get_bool(const gguf_context * g, const std::string & k, bool fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_bool(g, id);
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("gguf: missing required tensor '" + name + "'");
    return t;
}

std::vector<float> read_filterbank_to_vector(ggml_tensor * t) {
    const size_t n_elts = ggml_nelements(t);
    std::vector<float> out(n_elts);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n_elts * sizeof(float));
    } else {
        throw std::runtime_error("preproc tensor type must be f32");
    }
    return out;
}

}

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose) {
    (void) n_threads;
    (void) n_gpu_layers;

    auto impl = std::make_shared<ParakeetCtcModel::Impl>();

    gguf_init_params params = { false, &impl->ctx };
    impl->gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!impl->gguf) {
        std::fprintf(stderr, "gguf: failed to open %s\n", gguf_path.c_str());
        return 1;
    }

    gguf_context * g = impl->gguf;

    {
        const int id = find_key(g, "general.architecture");
        if (id < 0) {
            std::fprintf(stderr, "gguf: missing general.architecture\n");
            return 2;
        }
        const char * arch = gguf_get_val_str(g, id);
        if (std::strcmp(arch, "parakeet-ctc") != 0) {
            std::fprintf(stderr, "gguf: expected arch=parakeet-ctc, got '%s'\n", arch);
            return 2;
        }
    }

    out_model.encoder_cfg.d_model        = get_u32(g, "parakeet.encoder.d_model", 1024);
    out_model.encoder_cfg.n_layers       = get_u32(g, "parakeet.encoder.n_layers", 24);
    out_model.encoder_cfg.n_heads        = get_u32(g, "parakeet.encoder.n_heads", 8);
    out_model.encoder_cfg.head_dim       = get_u32(g, "parakeet.encoder.head_dim",
                                                   out_model.encoder_cfg.d_model / out_model.encoder_cfg.n_heads);
    out_model.encoder_cfg.ff_dim         = get_u32(g, "parakeet.encoder.ff_dim", 4096);
    out_model.encoder_cfg.conv_kernel    = get_u32(g, "parakeet.encoder.conv_kernel", 9);
    out_model.encoder_cfg.subsampling_factor    = get_u32(g, "parakeet.encoder.subsampling_factor", 8);
    out_model.encoder_cfg.subsampling_channels  = get_u32(g, "parakeet.encoder.subsampling_conv_channels", 256);
    out_model.encoder_cfg.subsampling_freq_bins = get_u32(g, "parakeet.encoder.subsampling_freq_bins", 10);
    out_model.encoder_cfg.pos_emb_max_len = get_u32(g, "parakeet.encoder.pos_emb_max_len", 5000);
    out_model.encoder_cfg.xscaling        = get_bool(g, "parakeet.encoder.xscaling", true);
    out_model.encoder_cfg.untie_biases    = get_bool(g, "parakeet.encoder.untie_biases", true);

    out_model.mel_cfg.sample_rate = get_u32(g, "parakeet.preproc.sample_rate", 16000);
    out_model.mel_cfg.n_fft       = get_u32(g, "parakeet.preproc.n_fft",       512);
    out_model.mel_cfg.win_length  = get_u32(g, "parakeet.preproc.win_length",  400);
    out_model.mel_cfg.hop_length  = get_u32(g, "parakeet.preproc.hop_length",  160);
    out_model.mel_cfg.n_mels      = get_u32(g, "parakeet.preproc.n_mels",      80);
    out_model.mel_cfg.preemph     = get_f32(g, "parakeet.preproc.preemph",     0.97f);
    out_model.mel_cfg.log_zero_guard_value =
        get_f32(g, "parakeet.preproc.log_zero_guard_value", 5.96046448e-08f);

    out_model.vocab_size = get_u32(g, "parakeet.ctc.vocab_size", 1025);
    out_model.blank_id   = get_u32(g, "parakeet.ctc.blank_id",   1024);
    out_model.vocab.blank_id = out_model.blank_id;

    {
        const int id = find_key(g, "tokenizer.ggml.tokens");
        if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_STRING) {
            const size_t n = gguf_get_arr_n(g, id);
            out_model.vocab.pieces.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const char * s = gguf_get_arr_str(g, id, i);
                if (s) out_model.vocab.pieces[i] = s;
            }
        }
        const int id_sc = find_key(g, "tokenizer.ggml.scores");
        if (id_sc >= 0 && gguf_get_arr_n(g, id_sc) == out_model.vocab.pieces.size()) {
            const float * p = (const float *) gguf_get_arr_data(g, id_sc);
            out_model.vocab.scores.assign(p, p + out_model.vocab.pieces.size());
        }
        const int id_tp = find_key(g, "tokenizer.ggml.token_type");
        if (id_tp >= 0 && gguf_get_arr_n(g, id_tp) == out_model.vocab.pieces.size()) {
            const int8_t * p = (const int8_t *) gguf_get_arr_data(g, id_tp);
            out_model.vocab.piece_types.assign(p, p + out_model.vocab.pieces.size());
        }
        out_model.vocab.unk_id = (int32_t) get_u32(g, "tokenizer.ggml.unk_token_id", (uint32_t) -1);
        out_model.vocab.bos_id = (int32_t) get_u32(g, "tokenizer.ggml.bos_token_id", (uint32_t) -1);
        out_model.vocab.eos_id = (int32_t) get_u32(g, "tokenizer.ggml.eos_token_id", (uint32_t) -1);
        out_model.vocab.pad_id = (int32_t) get_u32(g, "tokenizer.ggml.pad_token_id", (uint32_t) -1);
    }

    out_model.mel_filterbank = require_tensor(impl->ctx, "preproc.mel_filterbank");
    out_model.window         = require_tensor(impl->ctx, "preproc.window");

    out_model.mel_cfg.filterbank = read_filterbank_to_vector(out_model.mel_filterbank);
    out_model.mel_cfg.window     = read_filterbank_to_vector(out_model.window);

    out_model.subsampling.conv0_w    = require_tensor(impl->ctx, "encoder.subsampling.conv0.weight");
    out_model.subsampling.conv0_b    = require_tensor(impl->ctx, "encoder.subsampling.conv0.bias");
    out_model.subsampling.conv1_dw_w = require_tensor(impl->ctx, "encoder.subsampling.conv1_dw.weight");
    out_model.subsampling.conv1_dw_b = require_tensor(impl->ctx, "encoder.subsampling.conv1_dw.bias");
    out_model.subsampling.conv1_pw_w = require_tensor(impl->ctx, "encoder.subsampling.conv1_pw.weight");
    out_model.subsampling.conv1_pw_b = require_tensor(impl->ctx, "encoder.subsampling.conv1_pw.bias");
    out_model.subsampling.conv2_dw_w = require_tensor(impl->ctx, "encoder.subsampling.conv2_dw.weight");
    out_model.subsampling.conv2_dw_b = require_tensor(impl->ctx, "encoder.subsampling.conv2_dw.bias");
    out_model.subsampling.conv2_pw_w = require_tensor(impl->ctx, "encoder.subsampling.conv2_pw.weight");
    out_model.subsampling.conv2_pw_b = require_tensor(impl->ctx, "encoder.subsampling.conv2_pw.bias");
    out_model.subsampling.out_w      = require_tensor(impl->ctx, "encoder.subsampling.out.weight");
    out_model.subsampling.out_b      = require_tensor(impl->ctx, "encoder.subsampling.out.bias");

    out_model.blocks.resize(out_model.encoder_cfg.n_layers);
    for (int i = 0; i < out_model.encoder_cfg.n_layers; ++i) {
        BlockWeights & b = out_model.blocks[i];
        const std::string p = "encoder.blk." + std::to_string(i) + ".";

        b.norm_ff1_w  = require_tensor(impl->ctx, p + "norm_ff1.weight");
        b.norm_ff1_b  = require_tensor(impl->ctx, p + "norm_ff1.bias");
        b.ff1_l1_w    = require_tensor(impl->ctx, p + "ff1.linear1.weight");
        b.ff1_l1_b    = require_tensor(impl->ctx, p + "ff1.linear1.bias");
        b.ff1_l2_w    = require_tensor(impl->ctx, p + "ff1.linear2.weight");
        b.ff1_l2_b    = require_tensor(impl->ctx, p + "ff1.linear2.bias");

        b.norm_attn_w = require_tensor(impl->ctx, p + "norm_attn.weight");
        b.norm_attn_b = require_tensor(impl->ctx, p + "norm_attn.bias");
        b.attn_q_w    = require_tensor(impl->ctx, p + "attn.q.weight");
        b.attn_q_b    = require_tensor(impl->ctx, p + "attn.q.bias");
        b.attn_k_w    = require_tensor(impl->ctx, p + "attn.k.weight");
        b.attn_k_b    = require_tensor(impl->ctx, p + "attn.k.bias");
        b.attn_v_w    = require_tensor(impl->ctx, p + "attn.v.weight");
        b.attn_v_b    = require_tensor(impl->ctx, p + "attn.v.bias");
        b.attn_out_w  = require_tensor(impl->ctx, p + "attn.out.weight");
        b.attn_out_b  = require_tensor(impl->ctx, p + "attn.out.bias");
        b.attn_pos_w  = require_tensor(impl->ctx, p + "attn.pos.weight");
        b.pos_bias_u  = require_tensor(impl->ctx, p + "attn.pos_bias_u");
        b.pos_bias_v  = require_tensor(impl->ctx, p + "attn.pos_bias_v");

        b.norm_conv_w = require_tensor(impl->ctx, p + "norm_conv.weight");
        b.norm_conv_b = require_tensor(impl->ctx, p + "norm_conv.bias");
        b.conv_pw1_w  = require_tensor(impl->ctx, p + "conv.pw1.weight");
        b.conv_pw1_b  = require_tensor(impl->ctx, p + "conv.pw1.bias");
        b.conv_dw_w   = require_tensor(impl->ctx, p + "conv.dw.weight");
        b.conv_dw_b   = require_tensor(impl->ctx, p + "conv.dw.bias");
        b.conv_bn_scale = require_tensor(impl->ctx, p + "conv.bn.scale");
        b.conv_bn_shift = require_tensor(impl->ctx, p + "conv.bn.shift");
        b.conv_pw2_w  = require_tensor(impl->ctx, p + "conv.pw2.weight");
        b.conv_pw2_b  = require_tensor(impl->ctx, p + "conv.pw2.bias");

        b.norm_ff2_w  = require_tensor(impl->ctx, p + "norm_ff2.weight");
        b.norm_ff2_b  = require_tensor(impl->ctx, p + "norm_ff2.bias");
        b.ff2_l1_w    = require_tensor(impl->ctx, p + "ff2.linear1.weight");
        b.ff2_l1_b    = require_tensor(impl->ctx, p + "ff2.linear1.bias");
        b.ff2_l2_w    = require_tensor(impl->ctx, p + "ff2.linear2.weight");
        b.ff2_l2_b    = require_tensor(impl->ctx, p + "ff2.linear2.bias");

        b.norm_out_w  = require_tensor(impl->ctx, p + "norm_out.weight");
        b.norm_out_b  = require_tensor(impl->ctx, p + "norm_out.bias");
    }

    out_model.ctc.w = require_tensor(impl->ctx, "ctc.decoder.weight");
    out_model.ctc.b = require_tensor(impl->ctx, "ctc.decoder.bias");

    impl->backend = ggml_backend_cpu_init();
    if (!impl->backend) {
        std::fprintf(stderr, "gguf: ggml_backend_cpu_init failed\n");
        return 10;
    }

    int resolved_threads = n_threads;
    if (resolved_threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        resolved_threads = hc > 0 ? (int) hc : 4;
    }
    ggml_backend_cpu_set_n_threads(impl->backend, resolved_threads);

    out_model.impl = impl;

    if (verbose) {
        print_model_summary(out_model);
        std::fprintf(stderr, "  backend: cpu  (threads=%d)\n", resolved_threads);
    }
    return 0;
}

void print_model_summary(const ParakeetCtcModel & m) {
    std::fprintf(stderr, "parakeet-ctc loaded:\n");
    std::fprintf(stderr, "  encoder: d_model=%d n_layers=%d n_heads=%d head_dim=%d ff_dim=%d conv_k=%d sub=%dx xscaling=%d untie=%d\n",
                 m.encoder_cfg.d_model, m.encoder_cfg.n_layers, m.encoder_cfg.n_heads,
                 m.encoder_cfg.head_dim, m.encoder_cfg.ff_dim, m.encoder_cfg.conv_kernel,
                 m.encoder_cfg.subsampling_factor,
                 (int) m.encoder_cfg.xscaling, (int) m.encoder_cfg.untie_biases);
    std::fprintf(stderr, "  preproc: sr=%d n_fft=%d win=%d hop=%d n_mels=%d preemph=%.2f log_guard=%.2e\n",
                 m.mel_cfg.sample_rate, m.mel_cfg.n_fft, m.mel_cfg.win_length,
                 m.mel_cfg.hop_length, m.mel_cfg.n_mels, m.mel_cfg.preemph,
                 (double) m.mel_cfg.log_zero_guard_value);
    std::fprintf(stderr, "  ctc:     vocab=%d blank=%d\n", m.vocab_size, m.blank_id);
    std::fprintf(stderr, "  tensors: filterbank=%ldx%ld window=%ld blocks=%zu\n",
                 (long) m.mel_filterbank->ne[0], (long) m.mel_filterbank->ne[1],
                 (long) m.window->ne[0], m.blocks.size());
}

namespace {

ggml_tensor * conv_bias_bcast(ggml_context * ctx, ggml_tensor * bias, int64_t C) {
    return ggml_reshape_4d(ctx, bias, 1, 1, C, 1);
}

ggml_tensor * apply_time_mask(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask) {
    return ggml_mul(ctx, x, mask);
}

ggml_tensor * subsampling_graph(ggml_context    * gctx,
                                ggml_tensor     * mel_in,
                                const SubsamplingWeights & S,
                                int               subsampling_channels,
                                int               /*d_model*/,
                                ggml_tensor     * mask_t0,
                                ggml_tensor     * mask_t1,
                                ggml_tensor     * mask_t2,
                                ggml_tensor     * mask_t3) {
    ggml_tensor * x = mel_in;

    x = apply_time_mask(gctx, x, mask_t0);
    x = ggml_conv_2d(gctx, S.conv0_w, x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv0_b, subsampling_channels));
    x = apply_time_mask(gctx, x, mask_t1);
    x = ggml_relu(gctx, x);

    x = apply_time_mask(gctx, x, mask_t1);
    x = ggml_conv_2d_dw(gctx, S.conv1_dw_w, x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_dw_b, subsampling_channels));
    x = apply_time_mask(gctx, x, mask_t2);
    x = ggml_conv_2d(gctx, S.conv1_pw_w, x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_pw_b, subsampling_channels));
    x = apply_time_mask(gctx, x, mask_t2);
    x = ggml_relu(gctx, x);

    x = apply_time_mask(gctx, x, mask_t2);
    x = ggml_conv_2d_dw(gctx, S.conv2_dw_w, x, 2, 2, 1, 1, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_dw_b, subsampling_channels));
    x = apply_time_mask(gctx, x, mask_t3);
    x = ggml_conv_2d(gctx, S.conv2_pw_w, x, 1, 1, 0, 0, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_pw_b, subsampling_channels));
    x = apply_time_mask(gctx, x, mask_t3);
    x = ggml_relu(gctx, x);

    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];

    x = ggml_permute(gctx, x, 0, 2, 1, 3);
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, W * C, H);

    x = ggml_mul_mat(gctx, S.out_w, x);
    x = ggml_add(gctx, x, S.out_b);
    return x;
}

int _conv_out_len(int L, int k, int s, int p) {
    return (L + 2 * p - k) / s + 1;
}

std::vector<float> compute_rel_pos_encoding(int T, int D) {
    const int L = 2 * T - 1;
    std::vector<float> pe((size_t) L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) {
        div_term[i] = std::exp(-((float)(2 * i) * log10000 / (float) D));
    }
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin( (float) i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos( (float) i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float) i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float) i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[(size_t) t * D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(size_t) (T - 1 + t) * D + d] = neg_pe[t][d];
    }
    return pe;
}

ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    if (p_front > 0) {
        ggml_tensor * head = ggml_view_4d(ctx, x, p_front, x->ne[1], x->ne[2], x->ne[3],
                                          x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 0);
    }
    if (p_back > 0) {
        ggml_tensor * tail = ggml_view_4d(ctx, y, p_back, y->ne[1], y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 0);
    }
    return y;
}

ggml_tensor * conv1d_via_matmul(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * kf32 = kernel->type == GGML_TYPE_F32
                       ? kernel
                       : ggml_cast(ctx, kernel, GGML_TYPE_F32);
    ggml_tensor * im2col = ggml_im2col(ctx, kf32, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kf32, kf32->ne[0] * kf32->ne[1], kf32->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kf32->ne[2], im2col->ne[2]);
}

ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * conformer_ff_graph(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * norm_w, ggml_tensor * norm_b,
                                 ggml_tensor * l1_w,  ggml_tensor * l1_b,
                                 ggml_tensor * l2_w,  ggml_tensor * l2_b,
                                 float eps) {
    x = layer_norm_affine(ctx, x, norm_w, norm_b, eps);
    x = ggml_add(ctx, ggml_mul_mat(ctx, l1_w, x), l1_b);
    x = ggml_silu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, l2_w, x), l2_b);
    return x;
}

ggml_tensor * rel_pos_mha_graph(ggml_context * ctx, ggml_tensor * xn,
                                ggml_tensor * pos_emb,
                                const BlockWeights & W,
                                int H, int HD, int T) {
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_q_w, xn), W.attn_q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_k_w, xn), W.attn_k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, W.attn_v_w, xn), W.attn_v_b);
    ggml_tensor * p = ggml_mul_mat(ctx, W.attn_pos_w, pos_emb);

    q = ggml_reshape_3d(ctx, q, HD, H, T);
    k = ggml_reshape_3d(ctx, k, HD, H, T);
    v = ggml_reshape_3d(ctx, v, HD, H, T);
    p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);

    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    ggml_tensor * u_bias = ggml_reshape_3d(ctx, W.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, W.pos_bias_v, HD, 1, H);
    ggml_tensor * q_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_v = ggml_add(ctx, q_perm, v_bias);

    ggml_tensor * ac = ggml_mul_mat(ctx, k_perm, q_u);
    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_v);

    ggml_tensor * bd_padded   = zero_pad_dim0(ctx, bd, 1, 0);
    ggml_tensor * bd_viewed   = ggml_reshape_3d(ctx, bd_padded, T, 2 * T, H);
    ggml_tensor * bd_sliced   = ggml_view_3d(ctx, bd_viewed, T, 2 * T - 1, H,
                                             bd_viewed->nb[1], bd_viewed->nb[2], bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced), 2 * T - 1, T, H);
    ggml_tensor * bd_final    = ggml_view_3d(ctx, bd_reshaped, T, T, H,
                                             bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    bd_final = ggml_cont(ctx, bd_final);

    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float) HD));
    ggml_tensor * attn = ggml_soft_max(ctx, scores);

    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v   = ggml_mul_mat(ctx, v_for_mm, attn);
    ggml_tensor * merged   = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat     = ggml_reshape_2d(ctx, merged, HD * H, T);

    return ggml_add(ctx, ggml_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);
}

ggml_tensor * conformer_conv_graph(ggml_context * ctx, ggml_tensor * xn,
                                   const BlockWeights & W,
                                   int d_model, int T, int conv_kernel) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, xn, 1, 0, 2, 3));

    xt = conv1d_via_matmul(ctx, W.conv_pw1_w, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, W.conv_pw1_b, 1, 2 * d_model));

    ggml_tensor * half1 = ggml_view_2d(ctx, xt, xt->ne[0], d_model,
                                       xt->nb[1], 0);
    ggml_tensor * half2 = ggml_view_2d(ctx, xt, xt->ne[0], d_model,
                                       xt->nb[1], (size_t) d_model * xt->nb[1]);
    xt = ggml_mul(ctx, ggml_cont(ctx, half1),
                  ggml_sigmoid(ctx, ggml_cont(ctx, half2)));

    const int pad = (conv_kernel - 1) / 2;
    xt = ggml_conv_1d_dw(ctx, W.conv_dw_w, xt, 1, pad, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, W.conv_dw_b, 1, d_model));

    xt = ggml_mul(ctx, xt, ggml_reshape_2d(ctx, W.conv_bn_scale, 1, d_model));
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, W.conv_bn_shift, 1, d_model));

    xt = ggml_silu(ctx, xt);

    xt = conv1d_via_matmul(ctx, W.conv_pw2_w, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, W.conv_pw2_b, 1, d_model));

    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

ggml_tensor * conformer_block_graph(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * pos_emb,
                                    const BlockWeights & W,
                                    int d_model, int H, int HD, int T,
                                    int conv_kernel, float eps) {
    ggml_tensor * residual = x;
    ggml_tensor * y = conformer_ff_graph(ctx, x,
                                         W.norm_ff1_w, W.norm_ff1_b,
                                         W.ff1_l1_w,   W.ff1_l1_b,
                                         W.ff1_l2_w,   W.ff1_l2_b, eps);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    residual = x;
    ggml_tensor * xn = layer_norm_affine(ctx, x, W.norm_attn_w, W.norm_attn_b, eps);
    y = rel_pos_mha_graph(ctx, xn, pos_emb, W, H, HD, T);
    x = ggml_add(ctx, residual, y);

    residual = x;
    xn = layer_norm_affine(ctx, x, W.norm_conv_w, W.norm_conv_b, eps);
    y = conformer_conv_graph(ctx, xn, W, d_model, T, conv_kernel);
    x = ggml_add(ctx, residual, y);

    residual = x;
    y = conformer_ff_graph(ctx, x,
                           W.norm_ff2_w, W.norm_ff2_b,
                           W.ff2_l1_w,   W.ff2_l1_b,
                           W.ff2_l2_w,   W.ff2_l2_b, eps);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    x = layer_norm_affine(ctx, x, W.norm_out_w, W.norm_out_b, eps);
    return x;
}

}

int run_subsampling(ParakeetCtcModel   & model,
                    const float        * mel,
                    int                  n_mel_frames,
                    int                  n_mels,
                    std::vector<float> & out_feats,
                    int                & out_n_frames) {
    if (!model.impl || !model.impl->backend) return -1;

    ggml_backend_t backend = model.impl->backend;
    const int C_sub = model.encoder_cfg.subsampling_channels;
    const int d_model = model.encoder_cfg.d_model;

    int mel_valid = 0;
    for (int t = 0; t < n_mel_frames; ++t) {
        bool nonzero = false;
        for (int m = 0; m < n_mels; ++m) {
            if (mel[(size_t) t * n_mels + m] != 0.0f) { nonzero = true; break; }
        }
        if (nonzero) mel_valid = t + 1;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;

    const int L0 = n_mel_frames;
    const int L1 = _conv_out_len(L0, 3, 2, 1);
    const int L2 = _conv_out_len(L1, 3, 2, 1);
    const int L3 = _conv_out_len(L2, 3, 2, 1);

    const int V0 = mel_valid;
    const int V1 = _conv_out_len(V0, 3, 2, 1);
    const int V2 = _conv_out_len(V1, 3, 2, 1);
    const int V3 = _conv_out_len(V2, 3, 2, 1);

    auto make_mask = [](int L, int V) {
        std::vector<float> m(L, 0.0f);
        for (int t = 0; t < L && t < V; ++t) m[t] = 1.0f;
        return m;
    };
    std::vector<float> m0 = make_mask(L0, V0);
    std::vector<float> m1 = make_mask(L1, V1);
    std::vector<float> m2 = make_mask(L2, V2);
    std::vector<float> m3 = make_mask(L3, V3);

    const size_t overhead = ggml_tensor_overhead() * (GGML_DEFAULT_GRAPH_SIZE + 64)
                          + ggml_graph_overhead();
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    ggml_context * gctx = ggml_init(gp);
    if (!gctx) return -2;

    ggml_tensor * mel_in  = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
    ggml_set_name(mel_in, "mel_in");
    ggml_tensor * mask_t0 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L0, 1, 1);
    ggml_tensor * mask_t1 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L1, 1, 1);
    ggml_tensor * mask_t2 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L2, 1, 1);
    ggml_tensor * mask_t3 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L3, 1, 1);
    ggml_set_name(mask_t0, "mask_t0");
    ggml_set_name(mask_t1, "mask_t1");
    ggml_set_name(mask_t2, "mask_t2");
    ggml_set_name(mask_t3, "mask_t3");

    ggml_tensor * out = subsampling_graph(gctx, mel_in, model.subsampling, C_sub, d_model,
                                          mask_t0, mask_t1, mask_t2, mask_t3);
    ggml_set_name(out, "sub_out");

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return -3;
    }

    ggml_backend_tensor_set(mel_in, mel, 0, (size_t) n_mels * L0 * sizeof(float));
    ggml_backend_tensor_set(mask_t0, m0.data(), 0, m0.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t1, m1.data(), 0, m1.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t2, m2.data(), 0, m2.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t3, m3.data(), 0, m3.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return -4;
    }

    const int H_out = (int) out->ne[1];
    const int W_out = (int) out->ne[0];
    out_feats.resize((size_t) W_out * H_out);
    ggml_backend_tensor_get(out, out_feats.data(), 0, out_feats.size() * sizeof(float));
    out_n_frames = H_out;

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    return 0;
}

int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out) {
    if (!model.impl || !model.impl->backend) return -1;

    ggml_backend_t backend = model.impl->backend;
    ParakeetCtcModel::Impl & impl = *model.impl;
    const EncoderConfig & enc = model.encoder_cfg;
    const int C_sub = enc.subsampling_channels;
    const int d_model = enc.d_model;
    const int H  = enc.n_heads;
    const int HD = enc.head_dim;
    const int N_LAYERS = enc.n_layers;
    const int conv_kernel = enc.conv_kernel;
    const float eps = enc.layer_norm_eps;

    int mel_valid = 0;
    for (int t = 0; t < n_mel_frames; ++t) {
        bool nonzero = false;
        for (int m = 0; m < n_mels; ++m) {
            if (mel[(size_t) t * n_mels + m] != 0.0f) { nonzero = true; break; }
        }
        if (nonzero) mel_valid = t + 1;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;

    const int L0 = n_mel_frames;
    const int L1 = _conv_out_len(L0, 3, 2, 1);
    const int L2 = _conv_out_len(L1, 3, 2, 1);
    const int L3 = _conv_out_len(L2, 3, 2, 1);

    const int V0 = mel_valid;
    const int V1 = _conv_out_len(V0, 3, 2, 1);
    const int V2 = _conv_out_len(V1, 3, 2, 1);
    const int V3 = _conv_out_len(V2, 3, 2, 1);

    const int T = L3;

    auto make_mask = [](int L, int V) {
        std::vector<float> m(L, 0.0f);
        for (int t = 0; t < L && t < V; ++t) m[t] = 1.0f;
        return m;
    };
    std::vector<float> m0 = make_mask(L0, V0);
    std::vector<float> m1 = make_mask(L1, V1);
    std::vector<float> m2 = make_mask(L2, V2);
    std::vector<float> m3 = make_mask(L3, V3);

    std::vector<float> pe_host = compute_rel_pos_encoding(T, d_model);

    const size_t graph_slots = GGML_DEFAULT_GRAPH_SIZE * 16;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    ggml_context * gctx = ggml_init(gp);
    if (!gctx) return -2;

    ggml_tensor * mel_in  = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
    ggml_tensor * mask_t0 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L0, 1, 1);
    ggml_tensor * mask_t1 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L1, 1, 1);
    ggml_tensor * mask_t2 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L2, 1, 1);
    ggml_tensor * mask_t3 = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, 1, L3, 1, 1);
    ggml_tensor * pe_in   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, 2 * T - 1);
    ggml_set_name(mel_in, "mel_in");
    ggml_set_name(mask_t0, "mask_t0");
    ggml_set_name(mask_t1, "mask_t1");
    ggml_set_name(mask_t2, "mask_t2");
    ggml_set_name(mask_t3, "mask_t3");
    ggml_set_name(pe_in,   "pe_in");

    ggml_tensor * x = subsampling_graph(gctx, mel_in, model.subsampling, C_sub, d_model,
                                        mask_t0, mask_t1, mask_t2, mask_t3);
    ggml_tensor * sub_out_node = x;
    ggml_set_name(sub_out_node, "subsampling_out");
    ggml_set_output(sub_out_node);

    x = ggml_scale(gctx, x, std::sqrt((float) d_model));

    ggml_tensor * block_0_out_node = nullptr;
    ggml_tensor * block_last_out_node = nullptr;

    const int n_run_layers = std::getenv("PARAKEET_MAX_LAYERS")
                           ? std::atoi(std::getenv("PARAKEET_MAX_LAYERS"))
                           : N_LAYERS;

    ggml_tensor * post_ff1_0_node  = nullptr;
    ggml_tensor * post_attn_0_node = nullptr;
    ggml_tensor * post_conv_0_node = nullptr;
    ggml_tensor * post_ff2_0_node  = nullptr;

    for (int i = 0; i < n_run_layers; ++i) {
        if (i == 0) {
            const BlockWeights & W = model.blocks[0];
            ggml_tensor * residual = x;
            ggml_tensor * y = conformer_ff_graph(gctx, x,
                                                 W.norm_ff1_w, W.norm_ff1_b,
                                                 W.ff1_l1_w,   W.ff1_l1_b,
                                                 W.ff1_l2_w,   W.ff1_l2_b, eps);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            post_ff1_0_node = x;
            ggml_set_name(post_ff1_0_node, "block_0_post_ff1");
            ggml_set_output(post_ff1_0_node);

            residual = x;
            ggml_tensor * xn = layer_norm_affine(gctx, x, W.norm_attn_w, W.norm_attn_b, eps);
            y = rel_pos_mha_graph(gctx, xn, pe_in, W, H, HD, T);
            x = ggml_add(gctx, residual, y);
            post_attn_0_node = x;
            ggml_set_name(post_attn_0_node, "block_0_post_attn");
            ggml_set_output(post_attn_0_node);

            residual = x;
            xn = layer_norm_affine(gctx, x, W.norm_conv_w, W.norm_conv_b, eps);
            y = conformer_conv_graph(gctx, xn, W, d_model, T, conv_kernel);
            x = ggml_add(gctx, residual, y);
            post_conv_0_node = x;
            ggml_set_name(post_conv_0_node, "block_0_post_conv");
            ggml_set_output(post_conv_0_node);

            residual = x;
            y = conformer_ff_graph(gctx, x,
                                   W.norm_ff2_w, W.norm_ff2_b,
                                   W.ff2_l1_w,   W.ff2_l1_b,
                                   W.ff2_l2_w,   W.ff2_l2_b, eps);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            post_ff2_0_node = x;
            ggml_set_name(post_ff2_0_node, "block_0_post_ff2");
            ggml_set_output(post_ff2_0_node);

            x = layer_norm_affine(gctx, x, W.norm_out_w, W.norm_out_b, eps);

            block_0_out_node = x;
            ggml_set_name(block_0_out_node, "block_0_out");
            ggml_set_output(block_0_out_node);
        } else {
            x = conformer_block_graph(gctx, x, pe_in, model.blocks[i],
                                      d_model, H, HD, T, conv_kernel, eps);
        }
        if (i == n_run_layers - 1) {
            block_last_out_node = x;
            ggml_set_name(block_last_out_node, "block_last_out");
            ggml_set_output(block_last_out_node);
        }
    }

    ggml_tensor * encoder_out_node = x;
    ggml_set_name(encoder_out_node, "encoder_out");
    ggml_set_output(encoder_out_node);

    ggml_tensor * logits_node = ggml_add(gctx, ggml_mul_mat(gctx, model.ctc.w, x), model.ctc.b);
    ggml_set_name(logits_node, "logits");
    ggml_set_output(logits_node);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_slots, false);
    ggml_build_forward_expand(gf, sub_out_node);
    if (post_ff1_0_node)     ggml_build_forward_expand(gf, post_ff1_0_node);
    if (post_attn_0_node)    ggml_build_forward_expand(gf, post_attn_0_node);
    if (post_conv_0_node)    ggml_build_forward_expand(gf, post_conv_0_node);
    if (post_ff2_0_node)     ggml_build_forward_expand(gf, post_ff2_0_node);
    if (block_0_out_node)    ggml_build_forward_expand(gf, block_0_out_node);
    if (block_last_out_node) ggml_build_forward_expand(gf, block_last_out_node);
    ggml_build_forward_expand(gf, encoder_out_node);
    ggml_build_forward_expand(gf, logits_node);

    if (!impl.encoder_alloc || impl.encoder_alloc_T_mel != n_mel_frames) {
        if (impl.encoder_alloc) ggml_gallocr_free(impl.encoder_alloc);
        impl.encoder_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        impl.encoder_alloc_T_mel = n_mel_frames;
    }
    if (!impl.encoder_alloc || !ggml_gallocr_alloc_graph(impl.encoder_alloc, gf)) {
        ggml_free(gctx);
        return -3;
    }

    ggml_backend_tensor_set(mel_in,  mel,           0, (size_t) n_mels * L0 * sizeof(float));
    ggml_backend_tensor_set(mask_t0, m0.data(),     0, m0.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t1, m1.data(),     0, m1.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t2, m2.data(),     0, m2.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t3, m3.data(),     0, m3.size() * sizeof(float));
    ggml_backend_tensor_set(pe_in,   pe_host.data(), 0, pe_host.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        return -4;
    }

    out.n_enc_frames = T;
    out.d_model      = d_model;
    out.vocab_size   = model.vocab_size;

    auto copy_tensor = [&](ggml_tensor * t, std::vector<float> & dst) {
        if (!t) { dst.clear(); return; }
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    copy_tensor(sub_out_node,         out.subsampling_out);
    copy_tensor(post_ff1_0_node,      out.block_0_post_ff1);
    copy_tensor(post_attn_0_node,     out.block_0_post_attn);
    copy_tensor(post_conv_0_node,     out.block_0_post_conv);
    copy_tensor(post_ff2_0_node,      out.block_0_post_ff2);
    copy_tensor(block_0_out_node,     out.block_0_out);
    copy_tensor(block_last_out_node,  out.block_last_out);
    copy_tensor(encoder_out_node,     out.encoder_out);
    copy_tensor(logits_node,          out.logits);

    ggml_free(gctx);
    return 0;
}

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id) {
    std::vector<int32_t> decoded;
    decoded.reserve(n_frames);

    int32_t prev = -1;
    for (int t = 0; t < n_frames; ++t) {
        const float * row = logits + static_cast<size_t>(t) * vocab_size;
        int32_t best       = 0;
        float   best_score = row[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (row[i] > best_score) { best_score = row[i]; best = i; }
        }
        if (best != blank_id && best != prev) decoded.push_back(best);
        prev = best;
    }

    return decoded;
}

}
