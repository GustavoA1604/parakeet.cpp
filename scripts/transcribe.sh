#!/usr/bin/env bash
# Thin convenience wrapper around ./build/parakeet that fills in the
# pinned model path.  Equivalent to:
#
#     ./build/parakeet --model models/parakeet-ctc-0.6b.gguf --wav "$1"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DEFAULT="${REPO_ROOT}/models/parakeet-ctc-0.6b.gguf"
BIN_DEFAULT="${REPO_ROOT}/build/parakeet"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <input.wav> [model.gguf] [parakeet binary]" >&2
    exit 2
fi

WAV="$1"
MODEL="${2:-$MODEL_DEFAULT}"
BIN="${3:-$BIN_DEFAULT}"

if [ ! -f "$MODEL" ]; then
    echo "error: model GGUF not found at $MODEL" >&2
    echo "  hint: run scripts/convert-nemo-to-gguf.py first" >&2
    exit 3
fi

exec "$BIN" --model "$MODEL" --wav "$WAV"
