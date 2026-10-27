#!/usr/bin/env bash
# Run the full MSK RFM-vs-XGBoost comparison, logging each step to logs/.
# Meant to run detached on the Lambda machine:  nohup bash run_lambda.sh > logs/all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

run() {
    echo "=== $(date '+%F %T')  $*"
    conda run --no-capture-output -n gfm python -u "$@" > "logs/${1%.py}.log" 2>&1 \
        && echo "    done: logs/${1%.py}.log" || echo "    FAILED: see logs/${1%.py}.log"
}

nvidia-smi --query-gpu=name,memory.total --format=csv
run msk_rfm.py              # RFM, 5-fold CV on training rows
run msk_rfm_eval.py         # RFM, train -> stage-1 development set
run msk_xgb_eval.py 100     # XGBoost, tuned on training rows -> stage-1
run msk_xgb.py 100          # XGBoost, nested-CV tuned, same 5 folds as msk_rfm.py
echo "=== $(date '+%F %T')  all done"
