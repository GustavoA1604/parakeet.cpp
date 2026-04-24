#pragma once

// Parakeet-TDT (Token-and-Duration Transducer) decoder.
//
// The TDT decoder runs on top of the existing Conformer encoder output
// exposed by run_encoder() (EncoderOutputs::encoder_out, shape [T_enc, D_enc]).
//
// Architecture (parakeet-tdt-0.6b-v3):
//   - Prediction network: 2-layer LSTM, hidden=640, vocab=8192 (+ blank_as_pad)
//   - Joint network:      enc_proj(1024->640) + pred_proj(640->640) + ReLU +
//                         out(640 -> 8192 + 1 blank + 5 durations = 8198)
//   - Greedy decode:      TDT loop that advances encoder frame pointer by the
//                         predicted duration each step (durations typically
//                         [0,1,2,3,4]), emits non-blank tokens, updates LSTM
//                         state on each non-blank emission.
//
// Weights live in the loaded ParakeetCtcModel (TdtWeights struct, see
// parakeet_ctc.h) as ggml_tensor pointers. They are dequantized into
// std::vector<float> once at Engine load time via tdt_prepare_runtime().

#include "parakeet_ctc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qvac_parakeet::ctc {

struct TdtRuntimeLstmLayer {
    std::vector<float> w_ih;
    std::vector<float> w_hh;
    std::vector<float> b_ih;
    std::vector<float> b_hh;
};

struct TdtRuntimeWeights {
    int H_pred  = 640;
    int H_joint = 640;
    int D_enc   = 1024;
    int V_plus_1 = 8193;
    int V_out   = 8198;
    int L       = 2;
    int num_durations = 5;

    std::vector<float> embed;
    std::vector<TdtRuntimeLstmLayer> lstm;

    std::vector<float> joint_enc_w;
    std::vector<float> joint_enc_b;
    std::vector<float> joint_pred_w;
    std::vector<float> joint_pred_b;
    std::vector<float> joint_out_w;
    std::vector<float> joint_out_b;
};

struct TdtDecodeOptions {
    int max_symbols_per_step = 10;
};

struct TdtDecodeResult {
    std::vector<int32_t> token_ids;
    std::string text;
    int steps = 0;
    double decode_ms = 0.0;
};

int tdt_prepare_runtime(const ParakeetCtcModel & model, TdtRuntimeWeights & out);

int tdt_greedy_decode(const ParakeetCtcModel & model,
                      const TdtRuntimeWeights & W,
                      const float * encoder_out,
                      int T_enc, int D_enc,
                      const TdtDecodeOptions & opts,
                      TdtDecodeResult & result);

}
