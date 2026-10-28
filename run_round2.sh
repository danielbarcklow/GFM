#!/usr/bin/env bash
# Round 2: label-sharing features again, now out-of-fold encoded (round 1 used
# leave-one-out, which leaks an inverted signal), plus a clean XGBoost baseline on
# the competition feature build. Selection metric: 5-fold CV on training rows.
# Run detached: setsid nohup bash run_round2.sh > logs/round2.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/bin:$PATH"
mkdir -p logs

(bash run_xgb_folds.sh 100 xgb_label2 --label-features > logs/xgb_label2_folds.log 2>&1
 bash run_xgb_folds.sh 100 xgb_comp --comp-features > logs/xgb_comp_folds.log 2>&1) &
XGB=$!

kumo() {  # kumo <name> [args...]
    echo "=== $(date '+%F %T')  start $1"
    conda run --no-capture-output -n gfm python -u msk_cv_kumo.py --name "$@" > "logs/cv_kumo_$1.log" 2>&1 \
        && echo "=== $(date '+%F %T')  done  $1" || echo "=== $(date '+%F %T')  FAILED $1"
}

kumo label2_pretrained   --steps 0   --label-features
kumo label2_ft200        --steps 200 --label-features
kumo label2_ft400_lr3e-6 --steps 400 --lr 3e-6 --label-features

wait "$XGB"
echo "=== $(date '+%F %T')  all done"
