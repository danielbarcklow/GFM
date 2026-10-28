# Final evaluation on the stage-2 (leaderboard) set, plus Kaggle submission files.
# Nothing is uploaded: submitting is a separate, manual step.
#
#   1. Label-shift adjustment check on stage-1: does the adjustment hurt when there
#      is (almost) no shift? Uses the saved stage-1 predictions.
#   2. Predict all 986 stage-2 rows with each model, using training + stage-1 rows
#      as labeled data (as competitors could).
#   3. Adjust each prediction for stage-2's unknown class mix, estimated from the
#      unlabeled stage-2 rows only (Saerens et al. 2002, EM).
#   4. Score once on the 125 labeled stage-2 rows, the set the final leaderboard used.
#   5. Write submission CSVs to data/msk/submissions/.
#
# Requires: data/msk/rfm_finetuned.pt   (msk_rfm_finetune.py)
#           data/msk/xgb_params.json    (msk_xgb_eval.py)
import json

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

import sdm

from msk_data import CLASSES, DATA_DIR, load_competition, score
from msk_rfm_eval import DEVICE, predict


def adjust_for_label_shift(proba, train_prior, iters=1000, tol=1e-8):
    """EM estimate of the test class mix from unlabeled predictions; returns (adjusted proba, mix)."""
    mix = train_prior.copy()
    for _ in range(iters):
        adjusted = proba * (mix / train_prior)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new_mix = adjusted.mean(axis=0)
        if np.abs(new_mix - mix).max() < tol:
            break
        mix = new_mix
    return adjusted, mix


def fmt(mix):
    return " ".join(f"{m:.2f}" for m in mix)


def main():
    variants, genes = load_competition()
    y = variants["Class"].fillna(0).astype(int).to_numpy()
    split = variants["split"].to_numpy()
    train, stage1, stage2 = split == "train", split == "stage1", split == "stage2"
    labeled2 = stage2 & (y > 0)
    train_prior = np.bincount(y[train], minlength=10)[1:] / train.sum()

    # 1. Does the adjustment hurt when train and test look alike (stage-1)?
    print("1. Label-shift adjustment on stage-1 (little shift expected):")
    for name, f in [("Kumo fine-tuned", "rfm_finetuned_stage1_proba.npy"), ("XGBoost", "xgb_stage1_proba.npy")]:
        p = np.load(DATA_DIR / f)
        adj, mix = adjust_for_label_shift(p, train_prior)
        print(f"   {name:16s} raw {score(y[stage1], p)['log_loss']:.4f}  ->  adjusted "
              f"{score(y[stage1], adj)['log_loss']:.4f}   (estimated mix: {fmt(mix)})")

    # 2. Predict all stage-2 rows; labeled data = training + stage-1.
    context = train | stage1
    preds = {}

    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    model.eval()
    preds["Kumo pretrained"] = predict(model, variants, genes, context, stage2)
    model.load_state_dict(torch.load(DATA_DIR / "rfm_finetuned.pt", map_location=DEVICE))
    model.eval()
    preds["Kumo fine-tuned"] = predict(model, variants, genes, context, stage2)

    cfg = json.loads((DATA_DIR / "xgb_params.json").read_text())
    X = variants.merge(genes, on="gene_id", how="left").drop(
        columns=["variant_id", "gene_id", "Class", "split", "kaggle_id"]
    )
    booster = xgb.train(cfg["params"], xgb.DMatrix(X[context], label=y[context] - 1),
                        num_boost_round=cfg["num_boost_round"])
    preds["XGBoost"] = booster.predict(xgb.DMatrix(X[stage2]))

    # 3 + 4. Adjust using all 986 unlabeled stage-2 rows; score on the 125 labeled ones.
    is_labeled = labeled2[stage2]
    rows = []
    out_dir = DATA_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)
    print(f"\n   training class mix:          {fmt(train_prior)}")
    for name, p in preds.items():
        adj, mix = adjust_for_label_shift(p, train_prior)
        print(f"   {name:16s} estimated stage-2 mix: {fmt(mix)}")
        for variant, q in [("raw", p), ("label-shift adjusted", adj)]:
            rows.append({"model": name, "probabilities": variant, **score(y[labeled2], q[is_labeled])})
            sub = pd.DataFrame(np.clip(q, 1e-6, 1), columns=[f"class{c}" for c in CLASSES])
            sub.insert(0, "ID", variants.loc[stage2, "kaggle_id"].to_numpy())
            fname = f"{name}_{variant}".replace(" ", "_").replace("-", "_").lower() + ".csv"
            sub.to_csv(out_dir / fname, index=False)

    res = pd.DataFrame(rows)
    print("\n4. Stage-2 leaderboard set (125 labeled rows), log loss lower is better:")
    print("   Leaderboard reference: 1st place 2.030, guessing 1/9 = 2.197")
    print(res.round(4).to_string(index=False))

    # Diagnostic only: it peeks at stage-2 labels, so it is NOT a valid result and
    # must not be used to pick a submission. It shows how much blending toward
    # "1/9 for every class" each model would have needed on this set.
    print("\n5. DIAGNOSTIC (uses stage-2 labels, not a valid result): best blend with uniform")
    uniform = np.full(len(CLASSES), 1 / len(CLASSES))
    for name, p in preds.items():
        q = p[is_labeled]
        best = min((score(y[labeled2], (1 - a) * q + a * uniform)["log_loss"], a)
                   for a in np.round(np.arange(0, 1.0001, 0.01), 2))
        print(f"   {name:16s} best weight on uniform {best[1]:.2f}  ->  log_loss {best[0]:.4f}")
    print(f"\nSubmission files (not uploaded): {out_dir}")


if __name__ == "__main__":
    main()
