#!/usr/bin/env bash
#
# Download every Parakeet checkpoint we know about into qvac-parakeet.cpp/models/
# (.nemo for the native ggml converter) and qvac/packages/qvac-lib-infer-parakeet/models/
# (ONNX bundles for the existing Node binding).
#
# Idempotent: skips files that already exist on disk. Re-run any time to top up.
# Total download budget on a clean machine: ~5 GiB at the time of writing
# (TDT v3 .nemo + TDT v3 ONNX bundle + EOU ONNX + Sortformer ONNX). Already-shipped
# CTC 0.6b / 1.1b checkpoints are untouched if present.
#
# Usage:
#     ./scripts/download-all-models.sh             # everything
#     ./scripts/download-all-models.sh nemo        # only .nemo files (native port targets)
#     ./scripts/download-all-models.sh onnx        # only ONNX bundles (binding targets)
#     ./scripts/download-all-models.sh tdt         # only the TDT v3 pair (.nemo + ONNX)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_DIR="$REPO_ROOT/models"
BINDING_DIR="${BINDING_DIR:-$REPO_ROOT/../qvac/packages/qvac-lib-infer-parakeet/models}"

mkdir -p "$NEMO_DIR"
mkdir -p "$BINDING_DIR"

want_nemo=1
want_onnx=1
case "${1:-all}" in
  all) ;;
  nemo) want_onnx=0 ;;
  onnx) want_nemo=0 ;;
  tdt)  ;;  # filtered below
  *) echo "usage: $0 [all|nemo|onnx|tdt]" >&2; exit 2 ;;
esac

bytes_human() {
  local b=$1
  if   (( b > 1<<30 )); then printf "%.2f GiB" "$(echo "$b / (1<<30)" | bc -l)"
  elif (( b > 1<<20 )); then printf "%.2f MiB" "$(echo "$b / (1<<20)" | bc -l)"
  else                       printf "%d B"     "$b"
  fi
}

fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    local sz; sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
    echo "  exists: $dest ($(bytes_human "$sz")) — skipping"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  fetching: $url"
  echo "          -> $dest"
  curl -L --fail --progress-bar -o "$dest.tmp" "$url"
  mv "$dest.tmp" "$dest"
  local sz; sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
  echo "  saved: $dest ($(bytes_human "$sz"))"
}

hr() { printf '%.0s=' {1..70}; echo; }

# -------------------- .nemo (native ggml port targets) --------------------
if (( want_nemo )); then
  if [[ "${1:-all}" == "tdt" ]] || [[ "${1:-all}" != "tdt" ]]; then
    hr
    echo "== nemo: parakeet-tdt-0.6b-v3 (multilingual, 25 langs, ~2.4 GiB)"
    fetch "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo" \
          "$NEMO_DIR/parakeet-tdt-0.6b-v3.nemo"
  fi

  if [[ "${1:-all}" != "tdt" ]]; then
    hr
    echo "== nemo: parakeet-ctc-0.6b (English, ~2.3 GiB)"
    fetch "https://huggingface.co/nvidia/parakeet-ctc-0.6b/resolve/main/parakeet-ctc-0.6b.nemo" \
          "$NEMO_DIR/parakeet-ctc-0.6b.nemo"

    hr
    echo "== nemo: parakeet-ctc-1.1b (English, ~4 GiB)"
    fetch "https://huggingface.co/nvidia/parakeet-ctc-1.1b/resolve/main/parakeet-ctc-1.1b.nemo" \
          "$NEMO_DIR/parakeet-ctc-1.1b.nemo"

    hr
    echo "== nemo: parakeet-tdt_ctc-110m (small TDT+CTC hybrid, ~440 MiB)"
    fetch "https://huggingface.co/nvidia/parakeet-tdt_ctc-110m/resolve/main/parakeet-tdt_ctc-110m.nemo" \
          "$NEMO_DIR/parakeet-tdt_ctc-110m.nemo"

    hr
    echo "== nemo: stt_en_fastconformer_hybrid_large_streaming_multi (EOU/streaming, ~440 MiB)"
    fetch "https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi/resolve/main/stt_en_fastconformer_hybrid_large_streaming_multi.nemo" \
          "$NEMO_DIR/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
  fi
fi

# -------------------- ONNX bundles (Node binding targets) --------------------
if (( want_onnx )); then
  hr
  echo "== onnx: parakeet-tdt-0.6b-v3 (binding 'tdt' slot, ~2.5 GiB)"
  TDT_DIR="$BINDING_DIR/parakeet-tdt-0.6b-v3-onnx"
  fetch "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx" \
        "$TDT_DIR/encoder-model.onnx"
  fetch "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data" \
        "$TDT_DIR/encoder-model.onnx.data"
  fetch "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx" \
        "$TDT_DIR/decoder_joint-model.onnx"
  fetch "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt" \
        "$TDT_DIR/vocab.txt"
  fetch "https://huggingface.co/ysdede/parakeet-tdt-0.6b-v2-onnx/resolve/main/nemo128.onnx" \
        "$TDT_DIR/preprocessor.onnx"

  if [[ "${1:-all}" != "tdt" ]]; then
    hr
    echo "== onnx: parakeet-ctc-0.6b (binding 'ctc' slot, ~2.4 GiB)"
    CTC_DIR="$BINDING_DIR/parakeet-ctc-0.6b-onnx"
    fetch "https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/resolve/main/onnx/model.onnx" \
          "$CTC_DIR/model.onnx"
    fetch "https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/resolve/main/onnx/model.onnx_data" \
          "$CTC_DIR/model.onnx_data"
    fetch "https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/resolve/main/tokenizer.json" \
          "$CTC_DIR/tokenizer.json"

    hr
    echo "== onnx: realtime_eou_120m-v1-onnx (binding 'eou' slot, ~200 MiB)"
    EOU_DIR="$BINDING_DIR/parakeet-eou-120m-v1-onnx"
    fetch "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/encoder.onnx" \
          "$EOU_DIR/encoder.onnx"
    fetch "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/decoder_joint.onnx" \
          "$EOU_DIR/decoder_joint.onnx"
    fetch "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/tokenizer.json" \
          "$EOU_DIR/tokenizer.json"

    hr
    echo "== onnx: diar_streaming_sortformer_4spk-v2 (binding 'sortformer' slot, ~100 MiB)"
    SORT_DIR="$BINDING_DIR/sortformer-4spk-v2-onnx"
    fetch "https://huggingface.co/cgus/diar_streaming_sortformer_4spk-v2-onnx/resolve/main/diar_streaming_sortformer_4spk-v2.onnx" \
          "$SORT_DIR/sortformer.onnx"
  fi
fi

hr
echo "done. Cached models:"
echo
echo "[.nemo (native ggml port)]"
ls -lh "$NEMO_DIR" | awk '/\.nemo$/ {print "  " $9, $5}' || true
echo
echo "[ONNX (Node binding)]"
find "$BINDING_DIR" -maxdepth 2 -name '*.onnx' -o -name '*.onnx_data' -o -name 'vocab.txt' -o -name 'tokenizer.json' 2>/dev/null \
    | sort | xargs -I{} sh -c 'echo "  $(ls -lh "{}" | awk "{print \$9, \$5}")"' 2>/dev/null || true
