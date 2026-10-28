#!/usr/bin/env bash
# Nested-CV XGBoost with all outer folds in parallel, each running parallel
# single-threaded trials on CPU, to fill every core of the machine.
# Usage: nohup bash run_xgb_folds.sh [n_trials] [tag] [extra msk_xgb.py args...] > logs/xgb_folds.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

N_TRIALS="${1:-100}"
TAG="${2:-xgb}"
shift $(( $# < 2 ? $# : 2 ))
EXTRA=("$@")
N_FOLDS=5
JOBS=$(( $(nproc) / N_FOLDS ))

rm -f logs/${TAG}_fold*.json
echo "=== $(date '+%F %T')  $N_FOLDS folds x $JOBS parallel trials, $N_TRIALS trials each"
for k in $(seq 0 $((N_FOLDS - 1))); do
    conda run --no-capture-output -n gfm python -u msk_xgb.py "$N_TRIALS" \
        --fold "$k" --device cpu --jobs "$JOBS" --tag "$TAG" "${EXTRA[@]}" > "logs/${TAG}_fold$k.log" 2>&1 &
done
wait
echo "=== $(date '+%F %T')  folds finished"
conda run --no-capture-output -n gfm python -u msk_xgb.py --collect --tag "$TAG"
echo "=== $(date '+%F %T')  all done"
