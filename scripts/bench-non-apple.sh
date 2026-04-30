#!/usr/bin/env bash
#
# bench-non-apple.sh — validate the Phase 14/15 TDT decoder Metal port
# (and the Phase 15.7 encoder QKV stack gate) on a non-Apple GPU
# backend.  All on-Apple measurements live in PROGRESS.md §14 / §15;
# this script captures the matching numbers on CUDA / Vulkan so the
# reviewer-flagged "non-Apple measurement is the blocker" gate clears
# without a chat-context handoff.
#
# What it does:
#   1. cmake-configures + builds two side-by-side trees:
#        build-cuda (   cmake -DGGML_CUDA=ON   )
#        build-vulkan ( cmake -DGGML_VULKAN=ON )
#      Skips a tree if the corresponding ggml backend isn't available
#      (no CUDA toolkit / no Vulkan SDK installed).
#   2. Runs test-tdt-decoder-parity on each available build (CPU vs
#      graph-path token-id parity gate).  Decoder is byte-exact
#      across all backends by design (parity test is the gate).
#   3. Runs qvac-parakeet --bench on each build, with --bench-warmup 3
#      --bench-runs 10 by default, and writes the JSON to
#      artifacts/bench/<backend>-phase15.json.
#   4. Pretty-prints the (decode, encoder, inference) means + stdevs
#      from each JSON for easy paste into PROGRESS.md.
#
# Usage:
#   scripts/bench-non-apple.sh [<gguf>] [<wav>] [-- extra flags ...]
# Defaults:
#   gguf = models/parakeet-tdt-0.6b-v3.q8_0.gguf
#   wav  = test/samples/sample-16k.wav
#
# Hand-off:
#   The reviewer / a Linux-CUDA box runs this script and scp's
#   artifacts/bench/*.json back; numbers go into PROGRESS.md §15.6 (TBD)
#   alongside the M3 Ultra row.

set -euo pipefail

GGUF=${1:-models/parakeet-tdt-0.6b-v3.q8_0.gguf}
WAV=${2:-test/samples/sample-16k.wav}
shift 2 2>/dev/null || true
EXTRA_FLAGS=("$@")

if [[ ! -f "$GGUF" ]]; then
    echo "missing $GGUF — run scripts/download-all-models.sh first" >&2
    exit 1
fi
if [[ ! -f "$WAV" ]]; then
    echo "missing $WAV" >&2
    exit 1
fi

mkdir -p artifacts/bench

build_one() {
    local backend=$1            # cuda | vulkan
    local cmake_flag=$2         # -DGGML_CUDA=ON | -DGGML_VULKAN=ON
    local build=build-${backend}

    echo
    echo "=== ${backend^^}: configure + build ==="
    if ! cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release $cmake_flag 2>&1 | tail -20; then
        echo "${backend}: cmake configure failed; skipping" >&2
        return 1
    fi
    if ! cmake --build "$build" --target qvac-parakeet test-tdt-decoder-parity \
            -j 2>&1 | tail -10; then
        echo "${backend}: build failed; skipping" >&2
        return 1
    fi
    return 0
}

run_one() {
    local backend=$1
    local build=build-${backend}
    local json=artifacts/bench/${backend}-phase15.json

    echo
    echo "=== ${backend^^}: parity gate ==="
    if ! "$build/test-tdt-decoder-parity" "$GGUF" "$WAV" 2>&1 \
            | grep -E "PASS|FAIL|all checks passed"; then
        echo "${backend}: parity gate did not pass; aborting" >&2
        return 1
    fi

    echo
    echo "=== ${backend^^}: bench ==="
    "$build/qvac-parakeet" \
        --bench --bench-json "$json" \
        --model "$GGUF" --wav "$WAV" \
        --n-gpu-layers 1 \
        --bench-warmup 3 --bench-runs 10 \
        "${EXTRA_FLAGS[@]}" 2>&1 | tail -25

    echo
    echo "JSON: $json"
}

summarise() {
    local backend=$1
    local json=artifacts/bench/${backend}-phase15.json
    [[ -f "$json" ]] || return 0
    python3 - "$json" "$backend" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
backend = sys.argv[2]
def pick(k):
    s = data.get(k + "_ms") or data.get(k)
    return s if isinstance(s, dict) else {"mean": s, "stdev": 0}
mel = pick("mel")
enc = pick("encoder")
dec = pick("decode")
inf = pick("inference")
def fmt(d): return f"{d['mean']:6.2f} ± {d.get('stdev', 0):4.2f}"
print(f"\n{backend.upper()} summary (mean ± stdev, ms):")
print(f"  mel        {fmt(mel)}")
print(f"  encoder    {fmt(enc)}")
print(f"  decode     {fmt(dec)}")
print(f"  inference  {fmt(inf)}")
PY
}

for backend in cuda vulkan; do
    case $backend in
        cuda)   flag=-DGGML_CUDA=ON ;;
        vulkan) flag=-DGGML_VULKAN=ON ;;
    esac
    if build_one "$backend" "$flag"; then
        run_one "$backend" || true
        summarise "$backend"
    fi
done

echo
echo "Done.  Compare the printed (decode, encoder, inference) means"
echo "against the M3 Ultra Metal row in PROGRESS.md §15.3:"
echo "  encoder    ~ 68.5 ms"
echo "  decode     ~ 42.4 ms (post-on-device-argmax)"
echo "  inference  ~ 125.9 ms"
echo
echo "Expected on a discrete GPU (per the on-device-argmax + QKV-stack"
echo "review-comment hypotheses): inference_ms drops vs the same backend"
echo "without the 32 KB-per-step logits readback and with the wider"
echo "Q8_0 mat-mul filling the SM grid.  Both deltas are predicted-"
echo "positive; this script is what makes them measurable."
