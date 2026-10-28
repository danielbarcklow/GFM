#!/usr/bin/env bash
# Round 4: variant-specific text features (sentences mentioning the row's mutation),
# on standard and gene-held-out CV, for Kumo and XGBoost. Waits for round 3.
# Run detached: setsid nohup bash run_round4.sh > logs/round4.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/bin:$PATH"
until grep -q "all done" logs/round3.log 2>/dev/null; do sleep 60; done

(bash run_xgb_folds.sh 100 xgb_vtext --variant-text > logs/xgb_vtext_folds.log 2>&1
 bash run_xgb_folds.sh 100 xgb_gene_vtext --variant-text --gene-folds > logs/xgb_gene_vtext_folds.log 2>&1) &
XGB=$!

kumo() {  # kumo <name> [args...]
    echo "=== $(date '+%F %T')  start $1"
    conda run --no-capture-output -n gfm python -u msk_cv_kumo.py --name "$@" > "logs/cv_kumo_$1.log" 2>&1 \
        && echo "=== $(date '+%F %T')  done  $1" || echo "=== $(date '+%F %T')  FAILED $1"
}

kumo vtext_pretrained          --steps 0                 --variant-text
kumo vtext_ft400_lr3e-6        --steps 400 --lr 3e-6     --variant-text
kumo gene_vtext_pretrained     --steps 0                 --variant-text --gene-folds
kumo gene_vtext_ft400_lr3e-6   --steps 400 --lr 3e-6     --variant-text --gene-folds

wait "$XGB"
echo "=== $(date '+%F %T')  all done"
