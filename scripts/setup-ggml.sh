#!/usr/bin/env bash
# Clone ggml into ./ggml at the commit this repo is pinned against, and
# apply every patch under patches/ in lexicographic order.  Idempotent:
# safe to re-run.
#
# Update GGML_COMMIT here whenever the pin is bumped; this file is the
# single source of truth for which upstream ggml qvac-parakeet.cpp builds
# against.  Mirrors the shape of chatterbox.cpp/scripts/setup-ggml.sh so
# that contributors familiar with that repo see a 1:1 equivalent here.
#
# Patches we ship today:
#   patches/ggml-opencl-allow-non-adreno.patch
#       Lets the ggml-opencl backend run on non-Adreno/Intel GPUs
#       (NVIDIA, AMD, Apple) so the build can be parity-tested on
#       commodity desktop hardware. Real Adreno deployments build with
#       the patch applied as a no-op (Adreno path is unchanged).
#   patches/ggml-opencl-program-binary-cache.patch
#       Persistent OpenCL kernel binary cache via clCreateProgramWithBinary +
#       CL_PROGRAM_BINARIES. Removes seconds of cold-start shader compile on
#       every Adreno / Mesa / Mali / iGPU launch by serialising compiled kernels
#       under $GGML_OPENCL_CACHE_DIR (or XDG/HOME fallback). Same shape as the
#       Vulkan pipeline-cache patch QVAC-17872 landed for chatterbox.cpp.
#       See patches/README.md for the full rationale.

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

# Find every patch under patches/ matching ggml-*.patch, sorted.
shopt -s nullglob
PATCHES=( "$REPO_ROOT"/patches/ggml-*.patch )
shopt -u nullglob

cd ggml

CURRENT="$(git rev-parse --short=8 HEAD 2>/dev/null || echo '')"
NEED_CHECKOUT="0"
if [ "$CURRENT" != "$GGML_COMMIT" ]; then
    NEED_CHECKOUT="1"
fi

if [ "$NEED_CHECKOUT" = "1" ]; then
    git checkout -- . 2>/dev/null || true
    git checkout "$GGML_COMMIT"
    echo "  -> ok, at $(git rev-parse --short=8 HEAD)"
fi

# Apply patches.  We always reset to the pinned commit before applying so
# this is fully idempotent: re-running the script never stacks patches on
# top of patches.  If a patch fails to apply we leave ggml/ on the pinned
# commit so the next attempt starts clean.
if [ ${#PATCHES[@]} -gt 0 ]; then
    if [ "$NEED_CHECKOUT" = "0" ]; then
        # Same commit as last run, but patches may already be applied;
        # reset to pristine before re-applying.
        if ! git diff --quiet || ! git diff --cached --quiet; then
            echo "  -> resetting ggml worktree to pristine ${GGML_COMMIT}"
            git checkout -- .
        fi
    fi
    for patch in "${PATCHES[@]}"; do
        echo "  -> applying $(basename "$patch")"
        if ! git apply --check "$patch" 2>/dev/null; then
            echo "    (already applied or merge conflict, skipping)"
            continue
        fi
        git apply "$patch"
    done
fi

echo
echo "ggml is ready. Next:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "    cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
