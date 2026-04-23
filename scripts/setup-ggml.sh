#!/usr/bin/env bash
# Clone ggml into ./ggml at the commit this repo is pinned against.
# Idempotent: safe to re-run.
#
# Update GGML_COMMIT here whenever the pin is bumped; this file is the
# single source of truth for which upstream ggml qvac-parakeet.cpp builds
# against.  Mirrors the shape of chatterbox.cpp/scripts/setup-ggml.sh so
# that contributors familiar with that repo see a 1:1 equivalent here.

set -euo pipefail

GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "qvac-parakeet.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  -> cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

cd ggml

CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
if [ "$CURRENT" = "$GGML_COMMIT" ]; then
    echo "  -> already at ${GGML_COMMIT}, nothing to do"
    exit 0
fi

git checkout -- . 2>/dev/null || true
git checkout "$GGML_COMMIT"

echo "  -> ok, at $(git rev-parse --short=8 HEAD)"
echo
echo "ggml is ready. Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
