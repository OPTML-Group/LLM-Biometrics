#!/usr/bin/env bash
# Scenario 3 -- shared base (Appendix B.3, Tables 2 / A3 / A7): 20 pairs.
#
# All pairs share a pretrained base and diverge only through post-training, so
# architecture and depth match and both metrics run. Trace saturates near 1
# here (0.997-1.000) -- Subspace Alignment is what separates these pairs, both
# by fine-tuning data scale and by post-training algorithm.
#
#   bash scripts/run_s3_shared_base.sh
#   J=1 bash scripts/run_s3_shared_base.sh              # Table A6, J ablation
#   NOISE_STD=0.0001 bash scripts/run_s3_shared_base.sh # Table A3/A7, noise column
#
# Environment: CACHE_DIR (default ~/.cache/llm-biometrics), RESULTS_DIR (default runs),
#              TOP_K (512), J (3), N_BOTTOM (3), NOISE_STD (0).
#
# Note: the four Alpaca data-scale lines in the pair list point at local SFT
# checkpoints. Edit configs/pairs/s3_shared_base.txt to match your paths, or
# they will be reported as failures and the rest of the sweep will continue.

set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_DIR="${CACHE_DIR:-${LLM_BIOMETRICS_CACHE:-$HOME/.cache/llm-biometrics}}"
RESULTS_DIR="${RESULTS_DIR:-runs}"
TOP_K="${TOP_K:-512}"
J="${J:-3}"
N_BOTTOM="${N_BOTTOM:-3}"
NOISE_STD="${NOISE_STD:-0}"

TAG=""
[[ "$J" != "3" ]] && TAG="${TAG}_J${J}"
[[ "$NOISE_STD" != "0" ]] && TAG="${TAG}_noise_${NOISE_STD}"

mkdir -p "$CACHE_DIR" "$RESULTS_DIR"

python -m llm_biometrics run-pairs \
    --pairs configs/pairs/s3_shared_base.txt \
    --method both \
    --components all \
    --top-k "$TOP_K" \
    -J "$J" \
    --n-bottom "$N_BOTTOM" \
    --noise-std "$NOISE_STD" \
    --cache-dir "$CACHE_DIR" \
    --results "$RESULTS_DIR/s3_shared_base${TAG}.json" \
    --multi-gpu

echo
echo "Expected: Trace ~1.0 for every pair; Subspace spread over ~0.2-0.99,"
echo "decreasing with post-training data scale and varying by algorithm."
