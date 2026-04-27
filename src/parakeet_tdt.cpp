#include "parakeet_tdt.h"
#include "sentencepiece_bpe.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace qvac_parakeet {

namespace {

int argmax_f32(const float * data, int n) {
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_val) { best_val = data[i]; best = i; }
    }
    return best;
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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

// Dequantise a GGUF tensor (f32, f16, q8_0, etc.) into a host float vector.
void dequantize_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (!t) throw std::runtime_error("tdt: missing tensor for host dequant");
    const size_t n = (size_t) ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
        return;
    }
    const auto * tr = ggml_get_type_traits(t->type);
    if (!tr || !tr->to_float) {
        throw std::runtime_error(std::string("tdt: no to_float for type ") +
                                 ggml_type_name(t->type));
    }
    tr->to_float(t->data, out.data(), (int64_t) n);
}

// ---- Scalar host-side LSTM step (CPU fallback) ----
void host_lstm_step(const TdtRuntimeWeights & W,
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
        const auto & w = W.host_lstm[layer];
        const float * h_l = h_state + (size_t) layer * H;
        float       * c_l = c_state + (size_t) layer * H;

        gemv_f32(w.w_ih.data(), x, w.b_ih.data(), scratch.data(), G, H);
        for (int i = 0; i < G; ++i) scratch[i] += w.b_hh[i];
        gemv_add_f32(w.w_hh.data(), h_l, scratch.data(), G, H);

        float * h_new = h_state + (size_t) layer * H;
        for (int i = 0; i < H; ++i) {
            const float i_g = sigmoidf(scratch[0 * H + i]);
            const float f_g = sigmoidf(scratch[1 * H + i]);
            const float g_g = std::tanh (scratch[2 * H + i]);
            const float o_g = sigmoidf(scratch[3 * H + i]);
            const float c_new = f_g * c_l[i] + i_g * g_g;
            c_l[i] = c_new;
            h_new[i] = o_g * std::tanh(c_new);
        }
        std::memcpy(layer_input.data(), h_new, (size_t) H * sizeof(float));
        x = layer_input.data();
    }
}

// ---- Scalar host-side joint step (CPU fallback) ----
//
// Recomputes joint_enc @ enc_frame per emission step. Profiling showed that
// hoisting this matmul to a full-window precompute regresses on CPU due to
// loss of cache locality for small (~250) windows; the original per-step
// path is faster on M-series CPUs.
void host_joint_step(const TdtRuntimeWeights & W,
                     const float * __restrict enc_frame,
                     const float * __restrict pred,
                     std::vector<float> & hidden,
                     std::vector<float> & logits) {
    const int H  = W.H_joint;
    const int Hp = W.H_pred;
    const int De = W.D_enc;
    const int Vo = W.V_out;

    hidden.resize(H);
    gemv_f32(W.host_joint_enc_w.data(), enc_frame, W.host_joint_enc_b.data(),
             hidden.data(), H, De);
    {
        std::vector<float> tmp(H);
        gemv_f32(W.host_joint_pred_w.data(), pred, W.host_joint_pred_b.data(),
                 tmp.data(), H, Hp);
        for (int i = 0; i < H; ++i) hidden[i] += tmp[i];
    }
    for (int i = 0; i < H; ++i) hidden[i] = std::max(0.0f, hidden[i]);

    logits.resize(Vo);
    gemv_f32(W.host_joint_out_w.data(), hidden.data(), W.host_joint_out_b.data(),
             logits.data(), Vo, H);
}

// Build a fixed-shape graph that runs `embed -> L-layer LSTM` for one
// emission step. Inputs are uploaded per call:
//   - token_in  : int32[1]          (token id; embed row is fetched on-device)
//   - h_in,c_in : f32[H, L]         (per-layer hidden / cell state, layer-major)
// Outputs (read back into TdtDecodeState):
//   - h_out,c_out : f32[H, L]
//   - pred_out    : f32[H]          (alias for the last layer's h_new)
void build_lstm_graph(TdtRuntimeWeights & rt) {
    const int H = rt.H_pred;
    const int L = rt.L;
    ggml_context * gctx = rt.gctx;

    rt.lstm_token_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    rt.lstm_h_in     = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, L);
    rt.lstm_c_in     = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, L);
    ggml_set_name(rt.lstm_token_in, "lstm.token_in");
    ggml_set_name(rt.lstm_h_in,     "lstm.h_in");
    ggml_set_name(rt.lstm_c_in,     "lstm.c_in");
    ggml_set_input(rt.lstm_token_in);
    ggml_set_input(rt.lstm_h_in);
    ggml_set_input(rt.lstm_c_in);

    // Embedding lookup. predict_embed has ne[0]=H, ne[1]=vocab+1; result is
    // [H, 1]. Reshape to [H] for the per-step LSTM input.
    ggml_tensor * x = ggml_get_rows(gctx, rt.weights->predict_embed, rt.lstm_token_in);
    x = ggml_reshape_1d(gctx, x, H);

    std::vector<ggml_tensor *> h_new_per_layer(L);
    std::vector<ggml_tensor *> c_new_per_layer(L);

    for (int l = 0; l < L; ++l) {
        const auto & w = rt.weights->lstm[l];
        ggml_tensor * h_l_in = ggml_view_1d(gctx, rt.lstm_h_in, H, (size_t) l * H * sizeof(float));
        ggml_tensor * c_l_in = ggml_view_1d(gctx, rt.lstm_c_in, H, (size_t) l * H * sizeof(float));

        // gates = w_ih @ x + b_ih + b_hh + w_hh @ h_prev   ->  [4H]
        ggml_tensor * gates = ggml_mul_mat(gctx, w.w_ih, x);
        gates = ggml_add(gctx, gates, w.b_ih);
        gates = ggml_add(gctx, gates, w.b_hh);
        ggml_tensor * gates_h = ggml_mul_mat(gctx, w.w_hh, h_l_in);
        gates = ggml_add(gctx, gates, gates_h);

        const size_t H_bytes = (size_t) H * sizeof(float);
        ggml_tensor * i_part = ggml_view_1d(gctx, gates, H, 0 * H_bytes);
        ggml_tensor * f_part = ggml_view_1d(gctx, gates, H, 1 * H_bytes);
        ggml_tensor * g_part = ggml_view_1d(gctx, gates, H, 2 * H_bytes);
        ggml_tensor * o_part = ggml_view_1d(gctx, gates, H, 3 * H_bytes);

        ggml_tensor * i_g = ggml_sigmoid(gctx, i_part);
        ggml_tensor * f_g = ggml_sigmoid(gctx, f_part);
        ggml_tensor * g_g = ggml_tanh   (gctx, g_part);
        ggml_tensor * o_g = ggml_sigmoid(gctx, o_part);

        // c_new = f * c_prev + i * g
        ggml_tensor * c_new = ggml_add(gctx,
                                        ggml_mul(gctx, f_g, c_l_in),
                                        ggml_mul(gctx, i_g, g_g));
        // h_new = o * tanh(c_new)
        ggml_tensor * h_new = ggml_mul(gctx, o_g, ggml_tanh(gctx, c_new));

        // ggml_concat / ggml_view_1d need contiguous tensors; some of the
        // intermediates above are views that ggml may not reshape in place.
        h_new = ggml_cont(gctx, h_new);
        c_new = ggml_cont(gctx, c_new);

        h_new_per_layer[l] = h_new;
        c_new_per_layer[l] = c_new;

        // Next layer feeds on this layer's hidden output.
        x = h_new;
    }

    // Stitch per-layer outputs back into a [H, L] tensor (matches input layout).
    ggml_tensor * h_out = h_new_per_layer[0];
    ggml_tensor * c_out = c_new_per_layer[0];
    if (L > 1) {
        h_out = ggml_reshape_2d(gctx, h_out, H, 1);
        c_out = ggml_reshape_2d(gctx, c_out, H, 1);
        for (int l = 1; l < L; ++l) {
            ggml_tensor * h_l = ggml_reshape_2d(gctx, h_new_per_layer[l], H, 1);
            ggml_tensor * c_l = ggml_reshape_2d(gctx, c_new_per_layer[l], H, 1);
            h_out = ggml_concat(gctx, h_out, h_l, 1);
            c_out = ggml_concat(gctx, c_out, c_l, 1);
        }
    }

    rt.lstm_h_out    = h_out;
    rt.lstm_c_out    = c_out;
    rt.lstm_pred_out = h_new_per_layer[L - 1];
    ggml_set_name(rt.lstm_h_out,    "lstm.h_out");
    ggml_set_name(rt.lstm_c_out,    "lstm.c_out");
    ggml_set_name(rt.lstm_pred_out, "lstm.pred_out");
    ggml_set_output(rt.lstm_h_out);
    ggml_set_output(rt.lstm_c_out);
    ggml_set_output(rt.lstm_pred_out);

    rt.g_lstm = ggml_new_graph_custom(gctx, /*size*/ 256, /*grads*/ false);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_h_out);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_c_out);
    ggml_build_forward_expand(rt.g_lstm, rt.lstm_pred_out);
}

// Build a fixed-shape graph that runs the joint network for one emission step.
// Inputs (uploaded per call):
//   - pred_in     : f32[H_pred]    (last LSTM hidden output)
//   - enc_proj_in : f32[H_joint]   (already includes joint_enc_b for this frame)
// Output:
//   - logits_out  : f32[V_out]
void build_joint_graph(TdtRuntimeWeights & rt) {
    const int H_pred  = rt.H_pred;
    const int H_joint = rt.H_joint;
    ggml_context * gctx = rt.gctx;

    rt.joint_pred_in     = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, H_pred);
    rt.joint_enc_proj_in = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, H_joint);
    ggml_set_name(rt.joint_pred_in,     "joint.pred_in");
    ggml_set_name(rt.joint_enc_proj_in, "joint.enc_proj_in");
    ggml_set_input(rt.joint_pred_in);
    ggml_set_input(rt.joint_enc_proj_in);

    // pred_proj = W_pred @ pred + b_pred
    ggml_tensor * pred_proj = ggml_mul_mat(gctx, rt.weights->joint_pred_w, rt.joint_pred_in);
    pred_proj = ggml_add(gctx, pred_proj, rt.weights->joint_pred_b);

    // hidden = relu(enc_proj_row + pred_proj)
    ggml_tensor * hidden = ggml_add(gctx, rt.joint_enc_proj_in, pred_proj);
    hidden = ggml_relu(gctx, hidden);

    // logits = W_out @ hidden + b_out
    ggml_tensor * logits = ggml_mul_mat(gctx, rt.weights->joint_out_w, hidden);
    logits = ggml_add(gctx, logits, rt.weights->joint_out_b);

    rt.joint_logits_out = logits;
    ggml_set_name(rt.joint_logits_out, "joint.logits");
    ggml_set_output(rt.joint_logits_out);

    rt.g_joint = ggml_new_graph_custom(gctx, /*size*/ 64, /*grads*/ false);
    ggml_build_forward_expand(rt.g_joint, rt.joint_logits_out);
}

// Build the full-window encoder-side projection graph for a given frame count.
// Topology:
//   enc[D_enc, T] -> joint_enc_w @ enc + joint_enc_b -> enc_proj[H_joint, T]
TdtRuntimeWeights::EncProjGraph build_enc_proj_graph(TdtRuntimeWeights & rt, int T) {
    TdtRuntimeWeights::EncProjGraph g{};
    g.T = T;

    const int H_joint = rt.H_joint;
    const int D_enc   = rt.D_enc;
    ggml_context * gctx = rt.gctx;

    g.enc_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, D_enc, T);
    ggml_set_name(g.enc_in, "enc_proj.enc_in");
    ggml_set_input(g.enc_in);

    ggml_tensor * proj = ggml_mul_mat(gctx, rt.weights->joint_enc_w, g.enc_in);
    proj = ggml_add(gctx, proj, rt.weights->joint_enc_b);

    g.out = proj;
    ggml_set_name(g.out, "enc_proj.out");
    ggml_set_output(g.out);

    g.cg = ggml_new_graph_custom(gctx, /*size*/ 32, /*grads*/ false);
    ggml_build_forward_expand(g.cg, g.out);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(rt.backend));
    if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.cg)) {
        std::fprintf(stderr, "tdt: failed to allocate enc_proj graph for T=%d\n", T);
        if (g.alloc) ggml_gallocr_free(g.alloc);
        g.alloc = nullptr;
    }

    (void) H_joint;
    return g;
}

const TdtRuntimeWeights::EncProjGraph * get_enc_proj_graph(TdtRuntimeWeights & rt, int T) {
    for (auto & g : rt.enc_proj_cache) {
        if (g.T == T) return &g;
    }
    if (rt.enc_proj_cache.size() >= TdtRuntimeWeights::k_enc_proj_cache_max) {
        // Evict the oldest cached entry.
        auto & victim = rt.enc_proj_cache.front();
        if (victim.alloc) ggml_gallocr_free(victim.alloc);
        rt.enc_proj_cache.erase(rt.enc_proj_cache.begin());
    }
    rt.enc_proj_cache.push_back(build_enc_proj_graph(rt, T));
    return &rt.enc_proj_cache.back();
}

bool compute_graph(TdtRuntimeWeights & rt, ggml_cgraph * cg) {
    if (rt.n_threads > 0 && ggml_backend_is_cpu(rt.backend)) {
        ggml_backend_cpu_set_n_threads(rt.backend, rt.n_threads);
    }
    return ggml_backend_graph_compute(rt.backend, cg) == GGML_STATUS_SUCCESS;
}

}  // anonymous namespace

TdtRuntimeWeights::TdtRuntimeWeights(TdtRuntimeWeights && o) noexcept { *this = std::move(o); }

TdtRuntimeWeights & TdtRuntimeWeights::operator=(TdtRuntimeWeights && o) noexcept {
    if (this == &o) return *this;
    this->~TdtRuntimeWeights();

    H_pred       = o.H_pred;
    H_joint      = o.H_joint;
    D_enc        = o.D_enc;
    V_plus_1     = o.V_plus_1;
    V_out        = o.V_out;
    L            = o.L;
    num_durations = o.num_durations;
    weights      = o.weights;        o.weights = nullptr;
    backend      = o.backend;        o.backend = nullptr;
    n_threads    = o.n_threads;
    gctx         = o.gctx;           o.gctx = nullptr;
    g_lstm       = o.g_lstm;         o.g_lstm = nullptr;
    alloc_lstm   = o.alloc_lstm;     o.alloc_lstm = nullptr;
    lstm_token_in = o.lstm_token_in; o.lstm_token_in = nullptr;
    lstm_h_in    = o.lstm_h_in;      o.lstm_h_in = nullptr;
    lstm_c_in    = o.lstm_c_in;      o.lstm_c_in = nullptr;
    lstm_h_out   = o.lstm_h_out;     o.lstm_h_out = nullptr;
    lstm_c_out   = o.lstm_c_out;     o.lstm_c_out = nullptr;
    lstm_pred_out = o.lstm_pred_out; o.lstm_pred_out = nullptr;
    g_joint      = o.g_joint;        o.g_joint = nullptr;
    alloc_joint  = o.alloc_joint;    o.alloc_joint = nullptr;
    joint_pred_in     = o.joint_pred_in;     o.joint_pred_in = nullptr;
    joint_enc_proj_in = o.joint_enc_proj_in; o.joint_enc_proj_in = nullptr;
    joint_logits_out  = o.joint_logits_out;  o.joint_logits_out = nullptr;
    enc_proj_cache = std::move(o.enc_proj_cache);
    o.enc_proj_cache.clear();
    return *this;
}

TdtRuntimeWeights::~TdtRuntimeWeights() {
    for (auto & g : enc_proj_cache) {
        if (g.alloc) ggml_gallocr_free(g.alloc);
    }
    enc_proj_cache.clear();
    if (alloc_joint) { ggml_gallocr_free(alloc_joint); alloc_joint = nullptr; }
    if (alloc_lstm)  { ggml_gallocr_free(alloc_lstm);  alloc_lstm  = nullptr; }
    if (gctx)        { ggml_free(gctx);                gctx        = nullptr; }
    // backend is owned by ParakeetCtcModel::Impl; don't free here.
}

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & W) {
    W = TdtRuntimeWeights{};

    W.H_pred        = model.encoder_cfg.tdt_pred_hidden;
    W.H_joint       = model.encoder_cfg.tdt_joint_hidden;
    W.D_enc         = model.encoder_cfg.d_model;
    W.L             = model.encoder_cfg.tdt_pred_rnn_layers;
    W.num_durations = model.encoder_cfg.tdt_num_durations;
    W.V_plus_1      = (int) model.vocab_size + 1;
    W.V_out         = W.V_plus_1 + W.num_durations;

    W.weights = &model.tdt;
    W.backend = model.backend_active();
    if (!W.backend) {
        std::fprintf(stderr, "tdt_prepare_runtime: model has no active backend (call load_from_gguf first)\n");
        return 1;
    }

    // Defensive thread-count for the rare case where graphs run on a CPU
    // backend (today they don't; CPU goes through the scalar fallback below).
    {
        const unsigned hc = std::thread::hardware_concurrency();
        W.n_threads = hc > 0 ? (int) hc : 4;
    }

    if (!model.tdt.predict_embed || model.tdt.lstm.empty() || !model.tdt.joint_out_w) {
        std::fprintf(stderr, "tdt_prepare_runtime: GGUF is missing TDT tensors\n");
        return 2;
    }

    // Decide the implementation path. The per-step graph dispatch overhead on
    // the CPU backend (thread-pool wakeup x ~250 emission steps) regresses
    // ~6x vs. a hand-rolled scalar gemv loop, so CPU keeps the legacy path.
    // GPU backends (Metal / CUDA / Vulkan) win even with per-step dispatch
    // because of native quantised matmul and faster argmax / large gemvs.
    W.use_graphs = !ggml_backend_is_cpu(W.backend);

    if (!W.use_graphs) {
        // ---- CPU fallback: dequantise weights to host f32 ----
        dequantize_to_f32(model.tdt.predict_embed, W.embed);
        W.host_lstm.clear();
        W.host_lstm.resize(W.L);
        for (int l = 0; l < W.L; ++l) {
            dequantize_to_f32(model.tdt.lstm[l].w_ih, W.host_lstm[l].w_ih);
            dequantize_to_f32(model.tdt.lstm[l].w_hh, W.host_lstm[l].w_hh);
            dequantize_to_f32(model.tdt.lstm[l].b_ih, W.host_lstm[l].b_ih);
            dequantize_to_f32(model.tdt.lstm[l].b_hh, W.host_lstm[l].b_hh);
        }
        dequantize_to_f32(model.tdt.joint_enc_w,  W.host_joint_enc_w);
        dequantize_to_f32(model.tdt.joint_enc_b,  W.host_joint_enc_b);
        dequantize_to_f32(model.tdt.joint_pred_w, W.host_joint_pred_w);
        dequantize_to_f32(model.tdt.joint_pred_b, W.host_joint_pred_b);
        dequantize_to_f32(model.tdt.joint_out_w,  W.host_joint_out_w);
        dequantize_to_f32(model.tdt.joint_out_b,  W.host_joint_out_b);
        return 0;
    }

    // ---- GPU path: build ggml graphs against native GGUF weight tensors ----
    // ggml_context for graph nodes only (tensor metadata + cgraph storage).
    // Per-tensor data buffers come from the gallocrs.
    const size_t graph_slots = 1024;
    const size_t graph_mem = ggml_tensor_overhead() * graph_slots
                           + ggml_graph_overhead_custom(graph_slots, false) * 4
                           + 64 * 1024;
    ggml_init_params gp = {};
    gp.mem_size   = graph_mem;
    gp.mem_buffer = nullptr;
    gp.no_alloc   = true;
    W.gctx = ggml_init(gp);
    if (!W.gctx) {
        std::fprintf(stderr, "tdt_prepare_runtime: ggml_init failed\n");
        return 3;
    }

    build_lstm_graph(W);
    build_joint_graph(W);

    W.alloc_lstm  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    W.alloc_joint = ggml_gallocr_new(ggml_backend_get_default_buffer_type(W.backend));
    if (!W.alloc_lstm || !W.alloc_joint) {
        std::fprintf(stderr, "tdt_prepare_runtime: failed to create gallocrs\n");
        return 4;
    }
    if (!ggml_gallocr_alloc_graph(W.alloc_lstm, W.g_lstm) ||
        !ggml_gallocr_alloc_graph(W.alloc_joint, W.g_joint)) {
        std::fprintf(stderr, "tdt_prepare_runtime: failed to allocate fixed-shape graphs\n");
        return 5;
    }

    return 0;
}

namespace {

// Run the LSTM-step graph. Inputs/outputs are host buffers; the graph copies
// them to/from the backend each call.
//   - token_id : embedding row to fetch on-device
//   - h_state  : in/out [L*H]
//   - c_state  : in/out [L*H]
//   - pred_out : out   [H]
bool run_lstm_step(TdtRuntimeWeights & rt,
                   int token_id,
                   float * h_state,
                   float * c_state,
                   float * pred_out) {
    const int H = rt.H_pred;
    const int L = rt.L;

    const int32_t tok = (int32_t) token_id;
    ggml_backend_tensor_set(rt.lstm_token_in, &tok, 0, sizeof(int32_t));
    ggml_backend_tensor_set(rt.lstm_h_in, h_state, 0, (size_t) L * H * sizeof(float));
    ggml_backend_tensor_set(rt.lstm_c_in, c_state, 0, (size_t) L * H * sizeof(float));

    if (!compute_graph(rt, rt.g_lstm)) {
        std::fprintf(stderr, "tdt: LSTM graph compute failed\n");
        return false;
    }

    ggml_backend_tensor_get(rt.lstm_h_out,    h_state, 0, (size_t) L * H * sizeof(float));
    ggml_backend_tensor_get(rt.lstm_c_out,    c_state, 0, (size_t) L * H * sizeof(float));
    ggml_backend_tensor_get(rt.lstm_pred_out, pred_out, 0, (size_t) H * sizeof(float));
    return true;
}

bool run_joint_step(TdtRuntimeWeights & rt,
                    const float * pred,
                    const float * enc_proj_row,
                    float * logits_out) {
    const int H_pred  = rt.H_pred;
    const int H_joint = rt.H_joint;
    const int V_out   = rt.V_out;

    ggml_backend_tensor_set(rt.joint_pred_in,     pred,         0, (size_t) H_pred  * sizeof(float));
    ggml_backend_tensor_set(rt.joint_enc_proj_in, enc_proj_row, 0, (size_t) H_joint * sizeof(float));

    if (!compute_graph(rt, rt.g_joint)) {
        std::fprintf(stderr, "tdt: joint graph compute failed\n");
        return false;
    }

    ggml_backend_tensor_get(rt.joint_logits_out, logits_out, 0, (size_t) V_out * sizeof(float));
    return true;
}

bool run_enc_proj(TdtRuntimeWeights & rt,
                  const float * encoder_out,
                  int T,
                  std::vector<float> & enc_proj_host) {
    const int H_joint = rt.H_joint;
    const int D_enc   = rt.D_enc;

    const TdtRuntimeWeights::EncProjGraph * g = get_enc_proj_graph(rt, T);
    if (!g || !g->alloc) return false;

    ggml_backend_tensor_set(g->enc_in, encoder_out, 0, (size_t) T * D_enc * sizeof(float));

    if (!compute_graph(rt, g->cg)) {
        std::fprintf(stderr, "tdt: enc_proj graph compute failed\n");
        return false;
    }

    enc_proj_host.resize((size_t) T * H_joint);
    ggml_backend_tensor_get(g->out, enc_proj_host.data(), 0, enc_proj_host.size() * sizeof(float));
    return true;
}

}  // anonymous namespace

void tdt_init_state(TdtRuntimeWeights & W, int blank_id, TdtDecodeState & state) {
    const int H = W.H_pred;
    const int L = W.L;

    state.h_state.assign((size_t) L * H, 0.0f);
    state.c_state.assign((size_t) L * H, 0.0f);
    state.pred_out.assign(H, 0.0f);
    state.symbols_this_step = 0;
    state.carry_frames      = 0;

    if (W.use_graphs) {
        if (!run_lstm_step(W, blank_id, state.h_state.data(), state.c_state.data(),
                           state.pred_out.data())) {
            throw std::runtime_error("tdt_init_state: LSTM graph compute failed");
        }
    } else {
        std::vector<float> scratch;
        const float * embed_row = W.embed.data() + (size_t) blank_id * H;
        host_lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(), scratch);
        std::memcpy(state.pred_out.data(),
                    state.h_state.data() + (size_t) (L - 1) * H,
                    (size_t) H * sizeof(float));
    }

    state.initialized = true;
}

int tdt_decode_window(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out_window,
                      int n_frames, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeState & state,
                      std::vector<int32_t> & out_tokens,
                      int & out_steps) {
    out_steps = 0;
    if (D_enc != W.D_enc) {
        std::fprintf(stderr, "tdt_decode_window: encoder d_model mismatch (%d vs %d)\n",
                     D_enc, W.D_enc);
        return 1;
    }
    if (n_frames <= 0) return 0;

    if (!state.initialized) {
        tdt_init_state(W, (int) model.blank_id, state);
    }

    const int H_pred  = W.H_pred;
    const int H_joint = W.H_joint;
    const int V_p1    = W.V_plus_1;
    const int D_n     = W.num_durations;
    const int L       = W.L;
    const int blank   = (int) model.blank_id;
    const int V_out   = W.V_out;

    // GPU path uses a one-shot full-window encoder projection on the
    // backend (the matmul wins big on Metal). CPU path keeps the original
    // per-step gemv inside host_joint_step (better cache locality).
    std::vector<float> enc_proj;
    if (W.use_graphs) {
        if (!run_enc_proj(W, encoder_out_window, n_frames, enc_proj)) return 6;
    }

    std::vector<float> logits((size_t) V_out);
    std::vector<float> scratch_lstm;
    std::vector<float> scratch_joint_hidden;

    int t = 0;
    if (state.carry_frames > 0) {
        t = std::min(state.carry_frames, n_frames);
        state.carry_frames -= t;
    }

    while (t < n_frames) {
        if (W.use_graphs) {
            const float * enc_proj_row = enc_proj.data() + (size_t) t * H_joint;
            if (!run_joint_step(W, state.pred_out.data(), enc_proj_row, logits.data())) return 7;
        } else {
            const float * enc_frame = encoder_out_window + (size_t) t * D_enc;
            host_joint_step(W, enc_frame, state.pred_out.data(),
                            scratch_joint_hidden, logits);
        }
        ++out_steps;

        const int best_token   = argmax_f32(logits.data(), V_p1);
        const int best_dur_idx = argmax_f32(logits.data() + V_p1, D_n);
        const int best_dur     = model.tdt_durations.empty()
                                   ? best_dur_idx
                                   : model.tdt_durations[best_dur_idx];

        if (best_token == blank) {
            t += std::max(1, best_dur);
            state.symbols_this_step = 0;
            continue;
        }

        out_tokens.push_back((int32_t) best_token);

        if (W.use_graphs) {
            if (!run_lstm_step(W, best_token,
                               state.h_state.data(),
                               state.c_state.data(),
                               state.pred_out.data())) return 8;
        } else {
            const float * embed_row = W.embed.data() + (size_t) best_token * H_pred;
            host_lstm_step(W, embed_row, state.h_state.data(), state.c_state.data(),
                           scratch_lstm);
            std::memcpy(state.pred_out.data(),
                        state.h_state.data() + (size_t) (L - 1) * H_pred,
                        (size_t) H_pred * sizeof(float));
        }

        ++state.symbols_this_step;
        if (best_dur > 0 || state.symbols_this_step >= opts.max_symbols_per_step) {
            t += std::max(1, best_dur);
            state.symbols_this_step = 0;
        }
    }

    state.carry_frames = std::max(0, t - n_frames);
    return 0;
}

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result) {
    const auto t0 = std::chrono::steady_clock::now();

    TdtDecodeState state;
    result.token_ids.clear();
    result.token_ids.reserve(T_enc);

    if (int rc = tdt_decode_window(model, W, encoder_out, T_enc, D_enc,
                                   opts, state, result.token_ids, result.steps);
        rc != 0) {
        return rc;
    }

    result.text = detokenize(model.vocab, result.token_ids);
    const auto t1 = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    return 0;
}

}
