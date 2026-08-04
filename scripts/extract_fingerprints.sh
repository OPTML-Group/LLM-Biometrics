#!/usr/bin/env bash
# Step 1 -- build the fingerprint cache.
#
#   bash scripts/extract_fingerprints.sh <MODEL> [MORE MODELS...]
#   bash scripts/extract_fingerprints.sh --from-pairs configs/pairs/s3_shared_base.txt
#
# Extraction is the only expensive step: it reads every weight matrix once and
# writes a per-model fingerprint file. Every scenario script below reuses that
# cache, so a model appearing in ten pairs is still read from disk once.
#
# Environment:
#   CACHE_DIR   where fingerprints are written   (default: ~/.cache/llm-biometrics)
#   METHOD      trace | subspace | both          (default: both)
#   TOP_K       singular directions retained     (default: 512)
#   NOISE_STD   Gaussian weight perturbation     (default: 0, i.e. clean)
#
# Size warning: a trace fingerprint is a few kB, but subspace bases run to
# several GB per model. Point CACHE_DIR at scratch space, not at this repo.

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_DIR="${CACHE_DIR:-${LLM_BIOMETRICS_CACHE:-$HOME/.cache/llm-biometrics}}"
METHOD="${METHOD:-both}"
TOP_K="${TOP_K:-512}"
NOISE_STD="${NOISE_STD:-0}"

if [[ $# -eq 0 ]]; then
    sed -n '2,20p' "$0"
    exit 1
fi

# Collect the model list, either given directly or read from a pair list.
MODELS=()
if [[ "${1:-}" == "--from-pairs" ]]; then
    [[ -f "${2:-}" ]] || { echo "no such pair list: ${2:-}" >&2; exit 1; }
    while read -r model; do
        MODELS+=("$model")
    done < <(sed 's/#.*//' "$2" | awk 'NF==2 {print $1; print $2}' | sort -u)
else
    MODELS=("$@")
fi

mkdir -p "$CACHE_DIR"
echo "Extracting ${#MODELS[@]} models  ->  $CACHE_DIR  [method=$METHOD, k=$TOP_K, sigma=$NOISE_STD]"

for model in "${MODELS[@]}"; do
    echo
    echo "############ $model ############"
    python -m llm_biometrics extract \
        --model "$model" \
        --method "$METHOD" \
        --components all \
        --top-k "$TOP_K" \
        --noise-std "$NOISE_STD" \
        --cache-dir "$CACHE_DIR" \
        --multi-gpu
done

echo
echo "Cache ready in $CACHE_DIR/ -- scenario scripts will now run without touching model weights."
