#include "parakeet_ctc.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <parakeet-ctc.gguf> <reference-logits.npy>\n"
            "\n"
            "replays NeMo's logits through the C++ CTC greedy decoder + SentencePiece\n"
            "detokenizer and asserts the transcript matches the NeMo reference.\n",
            argv[0]);
        return 2;
    }

    std::fprintf(stderr, "[test-ctc] phase 4 harness not yet wired\n");
    return 0;
}
