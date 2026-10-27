# XGBoost on the MSK competition protocol, matching msk_rfm_eval.py (no Kaggle submission).
#   training data:   the training rows (same features as the RFM, genes table joined in)
#   tuning:          Optuna, scored by 5-fold CV log loss on the training rows only
#   development set: 368 labeled stage-1 test rows, scored once with the tuned model
#
# Usage: python msk_xgb_eval.py [n_trials]   (default 100)
import sys
import time

import numpy as np
import xgboost as xgb

import msk_xgb
from msk_data import DATA_DIR, load_competition, score

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def main():
    variants, genes = load_competition()
    X = variants.merge(genes, on="gene_id", how="left").drop(
        columns=["variant_id", "gene_id", "Class", "split", "kaggle_id"]
    )
    train = (variants["split"] == "train").to_numpy()
    dev = (variants["split"] == "stage1").to_numpy()
    y_train = variants.loc[train, "Class"].astype(int).to_numpy() - 1  # XGBoost wants 0..8
    y_dev = variants.loc[dev, "Class"].astype(int).to_numpy()

    start = time.time()
    msk_xgb.N_TRIALS = N_TRIALS
    params, num_rounds = msk_xgb.tune(X[train], y_train)
    print(f"tuned in {time.time() - start:.0f}s: rounds={num_rounds}", flush=True)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()}, flush=True)

    booster = xgb.train(params, xgb.DMatrix(X[train], label=y_train), num_boost_round=num_rounds)
    proba = booster.predict(xgb.DMatrix(X[dev]))
    np.save(DATA_DIR / "xgb_stage1_proba.npy", proba)

    s = score(y_dev, proba)
    print(f"\nXGBoost ({N_TRIALS} trials) on stage-1 (368 rows): "
          f"log_loss={s['log_loss']:.4f}  accuracy={s['accuracy']:.4f}  macro_auc={s['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
