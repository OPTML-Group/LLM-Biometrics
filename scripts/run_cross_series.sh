#!/usr/bin/env bash
# Intermediate cross-series relationships (Appendix F, Table A5): 4 pairs.
#
# Qwen2.5 vs Qwen3 -- same provider and design philosophy, but different
# architecture and pretraining corpus, so neither "same series" nor
# "independent origin" fits. Their Trace scores land between the two regime
# means, which is the evidence that Trace reads as a continuous relatedness
# signal rather than a hard three-way label.
#
#   bash scripts/run_cross_series.sh
#
# Environment: CACHE_DIR (default ~/.cache/llm-biometrics), RESULTS_DIR (default runs).

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_DIR="${CACHE_DIR:-${LLM_BIOMETRICS_CACHE:-$HOME/.cache/llm-biometrics}}"
RESULTS_DIR="${RESULTS_DIR:-runs}"

mkdir -p "$CACHE_DIR" "$RESULTS_DIR"

python -m llm_biometrics run-pairs \
    --pairs configs/pairs/cross_series.txt \
    --method trace \
    --components all \
    --cache-dir "$CACHE_DIR" \
    --results "$RESULTS_DIR/cross_series.json" \
    --multi-gpu

echo
echo "Expected: paper mean 0.479, between independent-origin (0.311)"
echo "and same-series (0.774)."
