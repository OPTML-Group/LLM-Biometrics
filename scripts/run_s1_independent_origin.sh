#!/usr/bin/env bash
# Scenario 1 -- independent origin (Appendix B.1, Table A1): 27 pairs.
#
# Models from different organizations with no shared base. Architectures and
# depths differ, so only Trace applies: each layer-wise energy curve is
# resampled to a common depth before correlation.
#
#   bash scripts/run_s1_independent_origin.sh
#   NOISE_STD=0.0001 bash scripts/run_s1_independent_origin.sh   # Table A1, noise column
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
    --pairs configs/pairs/s1_independent_origin.txt \
    --method trace \
    --components all \
    --noise-std "$NOISE_STD" \
    --cache-dir "$CACHE_DIR" \
    --results "$RESULTS_DIR/s1_independent_origin${TAG}.json" \
    --multi-gpu

echo
echo "Expected: low scores (paper mean 0.311) -- these pairs share no lineage."
