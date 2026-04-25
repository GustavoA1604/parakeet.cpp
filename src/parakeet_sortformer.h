#pragma once

// Sortformer (4-speaker offline diarization) decoder/head, running on CPU
// in f32 after dequantizing weights from the loaded GGUF once at Engine
// construction. Pipeline:
//
//   encoder_out (T, D_enc)
//     -> encoder_proj  : Linear(D_enc -> tf_d)
//     -> transformer   : N_tf_layers x post-LN block
//                        (multi-head self-attn -> residual+LN -> FFN -> residual+LN)
//     -> head          : ReLU -> first_hidden_to_hidden(tf_d -> tf_d)
//                        -> ReLU -> single_hidden_to_spks(tf_d -> num_spks)
//                        -> sigmoid
//   speaker_probs (T, num_spks) in [0, 1]

#include "parakeet_ctc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet::ctc {

struct SortformerTransformerRuntimeBlock {
    std::vector<float> attn_q_w, attn_q_b;
    std::vector<float> attn_k_w, attn_k_b;
    std::vector<float> attn_v_w, attn_v_b;
    std::vector<float> attn_o_w, attn_o_b;
    std::vector<float> ln1_w,    ln1_b;
    std::vector<float> ffn_in_w, ffn_in_b;
    std::vector<float> ffn_out_w, ffn_out_b;
    std::vector<float> ln2_w,    ln2_b;
};

struct SortformerRuntimeWeights {
    int D_enc       = 512;
    int tf_d        = 192;
    int tf_inner    = 768;
    int tf_n_heads  = 8;
    int tf_n_layers = 18;
    int num_spks    = 4;
    int head_dim    = 24;

    std::vector<float> proj_w, proj_b;
    std::vector<SortformerTransformerRuntimeBlock> blocks;

    std::vector<float> head_h2h_w, head_h2h_b;
    std::vector<float> head_h2s_w, head_h2s_b;
};

struct SortformerDiarizationOptions {
    float threshold = 0.5f;
};

struct SortformerSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

struct SortformerDiarizationResult {
    int n_frames     = 0;
    int num_spks     = 0;
    double frame_stride_s = 0.08;
    std::vector<float> speaker_probs;
    std::vector<SortformerSegment> segments;
    double decode_ms = 0.0;
};

int  sortformer_prepare_runtime(const ParakeetCtcModel & model, SortformerRuntimeWeights & out);

int  sortformer_diarize(const ParakeetCtcModel & model,
                        const SortformerRuntimeWeights & W,
                        const float * encoder_out,
                        int T_enc, int D_enc,
                        const SortformerDiarizationOptions & opts,
                        SortformerDiarizationResult & out);

}
