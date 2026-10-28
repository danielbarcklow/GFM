#!/usr/bin/env bash
# Round 3: gene-held-out CV (every gene's rows in one fold), a stand-in for the
# shift to stage-2, where models are near chance. Also reruns the standard-CV
# configs so their out-of-fold probabilities are saved for choosing hedging.
# Waits for round 2 to finish first.
# Run detached: setsid nohup bash run_round3.sh > logs/round3.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/bin:$PATH"
until grep -q "all done" logs/round2.log 2>/dev/null; do sleep 60; done

bash run_xgb_folds.sh 100 xgb_gene --comp-features --gene-folds > logs/xgb_gene_folds.log 2>&1 &
XGB=$!

kumo() {  # kumo <name> [args...]
    echo "=== $(date '+%F %T')  start $1"
    conda run --no-capture-output -n gfm python -u msk_cv_kumo.py --name "$@" > "logs/cv_kumo_$1.log" 2>&1 \
        && echo "=== $(date '+%F %T')  done  $1" || echo "=== $(date '+%F %T')  FAILED $1"
}

kumo gene_pretrained        --steps 0                 --gene-folds
kumo gene_ft200             --steps 200               --gene-folds
kumo gene_ft400_lr3e-6      --steps 400 --lr 3e-6     --gene-folds
kumo oof_pretrained         --steps 0
kumo oof_ft400_lr3e-6       --steps 400 --lr 3e-6

wait "$XGB"
echo "=== $(date '+%F %T')  all done"
