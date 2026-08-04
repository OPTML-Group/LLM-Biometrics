#!/usr/bin/env bash
# Scenario 2 -- same series (Appendix B.2, Table A2): 10 pairs.
#
# Same series and training pipeline, differing mainly in parameter scale
# (1.4x for Pythia-1B/1.4B up to 8x for Mistral-7B vs Mixtral-8x7B). Depths
# differ, so only Trace applies -- and separating S2 from S1 is the hardest
# coarse-grained task, where Trace reaches AUC 0.850 against 0.655 for PDF.
#
#   bash scripts/run_s2_same_series.sh
#   NOISE_STD=0.0001 bash scripts/run_s2_same_series.sh   # Table A2, noise column
#
# Environment: CACHE_DIR (default ~/.cache/llm-biometrics), RESULTS_DIR (default runs),
#              NOISE_STD (default 0).

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_DIR="${CACHE_DIR:-${LLM_BIOMETRICS_CACHE:-$HOME/.cache/llm-biometrics}}"
RESULTS_DIR="${RESULTS_DIR:-runs}"
NOISE_STD="${NOISE_STD:-0}"
TAG=$([[ "$NOISE_STD" == "0" ]] && echo "" || echo "_noise_${NOISE_STD}")

mkdir -p "$CACHE_DIR" "$RESULTS_DIR"

python -m llm_biometrics run-pairs \
    --pairs configs/pairs/s2_same_series.txt \
    --method trace \
    --components all \
    --noise-std "$NOISE_STD" \
    --cache-dir "$CACHE_DIR" \
    --results "$RESULTS_DIR/s2_same_series${TAG}.json" \
    --multi-gpu

echo
echo "Expected: mid-range scores (paper mean 0.774), clearly above S1."
