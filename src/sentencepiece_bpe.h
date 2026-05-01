#pragma once

// Minimal SentencePiece BPE detokenizer for Parakeet-CTC.
//
// The SentencePiece model file is embedded into the GGUF as
// `tokenizer.ggml.model_type = "sentencepiece"` + `tokenizer.ggml.pieces`
// + `tokenizer.ggml.scores` + `tokenizer.ggml.piece_types` arrays
// (same schema used by llama.cpp).  For CTC, we only need the reverse
// direction (id -> text), so this file implements just that: given a
// decoded sequence of token ids, emit the NeMo-equivalent transcription.
//
// Implementation in src/sentencepiece_bpe.cpp.

#include <cstdint>
#include <string>
#include <vector>

namespace parakeet {

struct BpeVocab {
    std::vector<std::string> pieces;
    std::vector<float>        scores;
    std::vector<int8_t>       piece_types;

    int32_t blank_id = -1;
    int32_t unk_id   = -1;
    int32_t bos_id   = -1;
    int32_t eos_id   = -1;
    int32_t pad_id   = -1;
};

std::string detokenize(const BpeVocab & vocab,
                       const std::vector<int32_t> & token_ids);

}
