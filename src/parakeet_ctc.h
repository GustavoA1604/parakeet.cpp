#pragma once

// Parakeet-CTC model: GGUF loader + FastConformer encoder graph + CTC head
// + greedy decoder.  Phase 1 wires the loader; phase 3 the encoder graph.
//
// Implementation in src/parakeet_ctc.cpp.

#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

namespace qvac_parakeet::ctc {

struct EncoderConfig {
    int d_model        = 1024;
    int n_layers       = 24;
    int n_heads        = 8;
    int head_dim       = 128;
    int ff_dim         = 4096;
    int conv_kernel    = 9;
    int subsampling_factor     = 8;
    int subsampling_channels   = 256;
    int subsampling_freq_bins  = 10;
    int pos_emb_max_len        = 5000;
    bool xscaling              = true;
    bool untie_biases          = true;
    bool use_bias              = true;
    float layer_norm_eps       = 1.0e-5f;

    int tdt_pred_hidden        = 640;
    int tdt_pred_rnn_layers    = 2;
    int tdt_joint_hidden       = 640;
    int tdt_num_durations      = 5;

    int  sortformer_num_spks   = 4;
    int  sortformer_fc_d_model = 512;
    int  sortformer_tf_d_model = 192;
    int  sortformer_tf_n_layers   = 18;
    int  sortformer_tf_n_heads    = 8;
    int  sortformer_tf_inner_size = 768;
    bool sortformer_tf_pre_ln  = false;
};

struct SubsamplingWeights {
    ggml_tensor * conv0_w    = nullptr;
    ggml_tensor * conv0_b    = nullptr;
    ggml_tensor * conv1_dw_w = nullptr;
    ggml_tensor * conv1_dw_b = nullptr;
    ggml_tensor * conv1_pw_w = nullptr;
    ggml_tensor * conv1_pw_b = nullptr;
    ggml_tensor * conv2_dw_w = nullptr;
    ggml_tensor * conv2_dw_b = nullptr;
    ggml_tensor * conv2_pw_w = nullptr;
    ggml_tensor * conv2_pw_b = nullptr;
    ggml_tensor * out_w      = nullptr;
    ggml_tensor * out_b      = nullptr;
};

struct BlockWeights {
    ggml_tensor * norm_ff1_w = nullptr;
    ggml_tensor * norm_ff1_b = nullptr;
    ggml_tensor * ff1_l1_w   = nullptr;
    ggml_tensor * ff1_l1_b   = nullptr;
    ggml_tensor * ff1_l2_w   = nullptr;
    ggml_tensor * ff1_l2_b   = nullptr;

    ggml_tensor * norm_attn_w = nullptr;
    ggml_tensor * norm_attn_b = nullptr;
    ggml_tensor * attn_q_w    = nullptr;
    ggml_tensor * attn_q_b    = nullptr;
    ggml_tensor * attn_k_w    = nullptr;
    ggml_tensor * attn_k_b    = nullptr;
    ggml_tensor * attn_v_w    = nullptr;
    ggml_tensor * attn_v_b    = nullptr;
    ggml_tensor * attn_qkv_w  = nullptr;
    ggml_tensor * attn_qkv_b  = nullptr;
    ggml_tensor * attn_out_w  = nullptr;
    ggml_tensor * attn_out_b  = nullptr;
    ggml_tensor * attn_pos_w  = nullptr;
    ggml_tensor * pos_bias_u  = nullptr;
    ggml_tensor * pos_bias_v  = nullptr;

    ggml_tensor * norm_conv_w = nullptr;
    ggml_tensor * norm_conv_b = nullptr;
    ggml_tensor * conv_pw1_w  = nullptr;
    ggml_tensor * conv_pw1_b  = nullptr;
    ggml_tensor * conv_dw_w   = nullptr;
    ggml_tensor * conv_dw_b   = nullptr;
    ggml_tensor * conv_bn_scale = nullptr;
    ggml_tensor * conv_bn_shift = nullptr;
    ggml_tensor * conv_pw2_w  = nullptr;
    ggml_tensor * conv_pw2_b  = nullptr;

    ggml_tensor * norm_ff2_w = nullptr;
    ggml_tensor * norm_ff2_b = nullptr;
    ggml_tensor * ff2_l1_w   = nullptr;
    ggml_tensor * ff2_l1_b   = nullptr;
    ggml_tensor * ff2_l2_w   = nullptr;
    ggml_tensor * ff2_l2_b   = nullptr;

    ggml_tensor * norm_out_w = nullptr;
    ggml_tensor * norm_out_b = nullptr;
};

struct CtcHeadWeights {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
};

struct TdtLstmLayer {
    ggml_tensor * w_ih = nullptr;
    ggml_tensor * w_hh = nullptr;
    ggml_tensor * b_ih = nullptr;
    ggml_tensor * b_hh = nullptr;
};

struct TdtWeights {
    ggml_tensor * predict_embed = nullptr;
    std::vector<TdtLstmLayer> lstm;

    ggml_tensor * joint_enc_w  = nullptr;
    ggml_tensor * joint_enc_b  = nullptr;
    ggml_tensor * joint_pred_w = nullptr;
    ggml_tensor * joint_pred_b = nullptr;
    ggml_tensor * joint_out_w  = nullptr;
    ggml_tensor * joint_out_b  = nullptr;
};

enum class ParakeetModelType {
    CTC,
    TDT,
    SORTFORMER,
};

struct SortformerTransformerBlock {
    ggml_tensor * attn_q_w  = nullptr;
    ggml_tensor * attn_q_b  = nullptr;
    ggml_tensor * attn_k_w  = nullptr;
    ggml_tensor * attn_k_b  = nullptr;
    ggml_tensor * attn_v_w  = nullptr;
    ggml_tensor * attn_v_b  = nullptr;
    ggml_tensor * attn_o_w  = nullptr;
    ggml_tensor * attn_o_b  = nullptr;
    ggml_tensor * ln1_w     = nullptr;
    ggml_tensor * ln1_b     = nullptr;
    ggml_tensor * ffn_in_w  = nullptr;
    ggml_tensor * ffn_in_b  = nullptr;
    ggml_tensor * ffn_out_w = nullptr;
    ggml_tensor * ffn_out_b = nullptr;
    ggml_tensor * ln2_w     = nullptr;
    ggml_tensor * ln2_b     = nullptr;
};

struct SortformerWeights {
    ggml_tensor * encoder_proj_w = nullptr;
    ggml_tensor * encoder_proj_b = nullptr;
    std::vector<SortformerTransformerBlock> transformer;
    ggml_tensor * head_h2h_w = nullptr;
    ggml_tensor * head_h2h_b = nullptr;
    ggml_tensor * head_h2s_w = nullptr;
    ggml_tensor * head_h2s_b = nullptr;
};

struct ParakeetCtcModel {
    ParakeetModelType model_type = ParakeetModelType::CTC;

    EncoderConfig encoder_cfg;
    MelConfig     mel_cfg;
    BpeVocab      vocab;

    int32_t blank_id   = 1024;
    int32_t vocab_size = 1025;

    bool supports_streaming = false;

    std::vector<int32_t> tdt_durations;

    SubsamplingWeights       subsampling;
    std::vector<BlockWeights> blocks;
    CtcHeadWeights            ctc;
    TdtWeights                tdt;
    SortformerWeights         sortformer;

    ggml_tensor * mel_filterbank = nullptr;
    ggml_tensor * window         = nullptr;

    struct Impl;
    std::shared_ptr<Impl> impl;
};

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose);

void print_model_summary(const ParakeetCtcModel & m);

int run_subsampling(ParakeetCtcModel   & model,
                    const float        * mel,
                    int                  n_mel_frames,
                    int                  n_mels,
                    std::vector<float> & out_feats,
                    int                & out_n_frames);

struct EncoderOutputs {
    std::vector<float> subsampling_out;
    std::vector<float> block_0_post_ff1;
    std::vector<float> block_0_post_attn;
    std::vector<float> block_0_post_conv;
    std::vector<float> block_0_post_ff2;
    std::vector<float> block_0_out;
    std::vector<float> block_last_out;
    std::vector<float> encoder_out;
    std::vector<float> logits;
    int n_enc_frames = 0;
    int d_model      = 0;
    int vocab_size   = 0;
};

int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out,
                int                  max_layers = -1);

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id);

void ctc_greedy_decode_window(const float * logits,
                              int           start_frame,
                              int           end_frame,
                              int           vocab_size,
                              int32_t       blank_id,
                              int32_t     & inout_prev_token,
                              std::vector<int32_t> & out_tokens,
                              std::vector<int>     * out_first_frame = nullptr);

struct BlockSubstageTimes {
    double ff1_ms  = 0.0;
    double attn_ms = 0.0;
    double conv_ms = 0.0;
    double ff2_ms  = 0.0;
    double norm_out_ms = 0.0;
    double block_full_ms = 0.0;
};

int profile_block_substages(ParakeetCtcModel & model,
                            int T_enc,
                            int warmup_runs,
                            int timed_runs,
                            BlockSubstageTimes & out);

}
