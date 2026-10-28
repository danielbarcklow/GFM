#!/usr/bin/env bash
# Overnight pipeline on the GPU machine. Runs detached:
#   setsid nohup bash run_overnight.sh > logs/overnight.log 2>&1 < /dev/null &
#   1. XGBoost tuned on training rows -> stage-1 (CPU, saves its settings)   } in parallel
#   2. Kumo fine-tuning, 3 seeds (GPU); seed 0 is the pre-chosen model       }
#   3. Final stage-2 evaluation + submission files (nothing is uploaded)
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/miniforge3/bin:$PATH"
mkdir -p logs

py() {  # py <log name> <script> [args...]
    local log="logs/$1.log"; shift
    echo "=== $(date '+%F %T')  start $*"
    conda run --no-capture-output -n gfm python -u "$@" > "$log" 2>&1 \
        && echo "=== $(date '+%F %T')  done  $* -> $log" \
        || echo "=== $(date '+%F %T')  FAILED $* -> $log"
}

py overnight_xgb_eval msk_xgb_eval.py 100 --device cpu --jobs 10 &
XGB=$!

py overnight_ft_seed0 msk_rfm_finetune.py --seed 0 --out rfm_finetuned
py overnight_ft_seed1 msk_rfm_finetune.py --seed 1 --out rfm_finetuned_s1
py overnight_ft_seed2 msk_rfm_finetune.py --seed 2 --out rfm_finetuned_s2

wait "$XGB"
py overnight_final msk_final.py
echo "=== $(date '+%F %T')  all done"
