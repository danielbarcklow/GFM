#!/usr/bin/env bash
# Round 1 of experiments (selection metric: 5-fold CV on training rows only).
#   GPU: Kumo configurations, one after another (results in logs/cv_kumo.jsonl)
#   CPU: XGBoost nested CV with label-sharing features, all folds in parallel
# Run detached: setsid nohup bash run_round1.sh > logs/round1.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/bin:$PATH"
mkdir -p logs

bash run_xgb_folds.sh 100 xgb_label --label-features > logs/xgb_label_folds.log 2>&1 &
XGB=$!

kumo() {  # kumo <name> [args...]
    echo "=== $(date '+%F %T')  start $1"
    conda run --no-capture-output -n gfm python -u msk_cv_kumo.py --name "$@" > "logs/cv_kumo_$1.log" 2>&1 \
        && echo "=== $(date '+%F %T')  done  $1" || echo "=== $(date '+%F %T')  FAILED $1"
}

kumo pretrained          --steps 0
kumo ft200               --steps 200
kumo label_pretrained    --steps 0   --label-features
kumo label_ft200         --steps 200 --label-features
kumo ft200_lr3e-5        --steps 200 --lr 3e-5
kumo ft400_lr3e-6        --steps 400 --lr 3e-6
kumo ft400               --steps 400
kumo ft200_est32         --steps 200 --estimators 32
kumo ft200_3seeds        --steps 200 --seeds 3

wait "$XGB"
echo "=== $(date '+%F %T')  all done"
