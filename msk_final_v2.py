# Pre-registered final evaluation, round 4 recipe. Every choice below was fixed from
# 5-fold CV on the training rows (logs/cv_kumo.jsonl, logs/*_folds.log) before this
# script was run; stage-1 and stage-2 are each scored once and nothing is tuned on them.
#
#   features:  competition build + variant-specific text (msk_features.add_variant_text)
#   Kumo:      fine-tuned from the pretrained weights, 400 steps, lr 3e-6, seed 0
#   XGBoost:   Optuna-tuned (100 trials, inner 5-fold CV) on the training rows
#   hedging:   none (out-of-fold predictions, standard and gene-held-out, chose none)
#
#   stage-1:   models fit on training rows, labeled context = training rows
#   stage-2:   models refit on training + stage-1 rows (stage-1 labels were public by
#              stage 2), labeled context = training + stage-1 rows
#
# Submission CSVs for all 986 stage-2 rows are written to data/msk/submissions/; nothing
# is uploaded (submitting is a separate, manual step).
import os
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import msk_rfm_finetune as ft
import msk_xgb
import sdm
from msk_data import CLASSES, DATA_DIR, SEED, load_competition, score
from msk_features import add_variant_text
from msk_rfm_eval import DEVICE, predict

STEPS, LR = 400, 3e-6
N_TRIALS = 100


def kumo(variants, genes, y, labeled, query, pretrained):
    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    model.load_state_dict(pretrained)
    torch.manual_seed(SEED)
    ft.finetune(model, variants, genes, np.flatnonzero(labeled), y, STEPS, LR, seed=SEED)
    return predict(model, variants, genes, labeled, query)


def main():
    # tune as in the CV runs: parallel single-threaded CPU trials
    msk_xgb.N_TRIALS, msk_xgb.DEVICE, msk_xgb.PARALLEL_TRIALS = N_TRIALS, "cpu", os.cpu_count()
    variants, genes = load_competition()
    variants = add_variant_text(variants)
    split = variants["split"].to_numpy()
    train, stage1, stage2 = split == "train", split == "stage1", split == "stage2"
    labeled2 = stage2 & variants["Class"].notna().to_numpy()
    y = variants["Class"].fillna(0).astype(int).to_numpy()  # 0 = unlabeled, never used

    base = sdm.models.KumoRelational(task="classification", device=DEVICE)
    pretrained = {k: v.clone() for k, v in base.state_dict().items()}
    del base

    X = variants.merge(genes, on="gene_id", how="left").drop(
        columns=["variant_id", "gene_id", "Class", "split", "kaggle_id"])
    start = time.time()
    params, rounds = msk_xgb.tune(X[train], y[train] - 1)
    print(f"XGBoost tuned: {rounds} rounds, depth {params['max_depth']}, eta {params['eta']:.3f} "
          f"({time.time() - start:.0f}s)", flush=True)

    def xgboost(fit, query):
        booster = xgb.train(params, xgb.DMatrix(X[fit], label=y[fit] - 1), num_boost_round=rounds)
        return booster.predict(xgb.DMatrix(X[query]))

    uniform = np.full(len(CLASSES), 1 / len(CLASSES))
    rows = []
    preds1 = {"Kumo fine-tuned + variant text": kumo(variants, genes, y, train, stage1, pretrained),
              "XGBoost + variant text": xgboost(train, stage1)}
    for name, p in preds1.items():
        rows.append({"set": "stage-1 (368)", "model": name, **score(y[stage1], p)})
    rows.append({"set": "stage-1 (368)", "model": "uniform 1/9", **score(y[stage1], np.tile(uniform, (stage1.sum(), 1)))})

    fit2 = train | stage1
    preds2 = {"Kumo fine-tuned + variant text": kumo(variants, genes, y, fit2, stage2, pretrained),
              "XGBoost + variant text": xgboost(fit2, stage2)}
    is_labeled = labeled2[stage2]
    out_dir = DATA_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    for name, p in preds2.items():
        rows.append({"set": "stage-2 (125)", "model": name, **score(y[labeled2], p[is_labeled])})
        sub = pd.DataFrame(np.clip(p, 1e-6, 1), columns=[f"class{c}" for c in CLASSES])
        sub.insert(0, "ID", variants.loc[stage2, "kaggle_id"].to_numpy())
        sub.to_csv(out_dir / ("v2_" + name.split()[0].lower() + ".csv"), index=False)
    rows.append({"set": "stage-2 (125)", "model": "uniform 1/9", **score(y[labeled2], np.tile(uniform, (labeled2.sum(), 1)))})

    print("\nPre-registered final results (log loss lower is better; "
          "stage-2 leaderboard: 1st place 2.030):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\nSubmission files (not uploaded): {out_dir}/v2_*.csv")


if __name__ == "__main__":
    main()
