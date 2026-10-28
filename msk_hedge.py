# Choose hedging from out-of-fold predictions on the training rows only (never on
# stage-1 or stage-2). Two forms:
#   blend:       p = (1 - alpha) * p_model + alpha * q,  q = uniform or training class mix
#   temperature: p ∝ p_model ** t   (t < 1 flattens)
# Gene-held-out folds stand in for the shift to new data; standard folds are shown
# for comparison.
#
# Usage: python msk_hedge.py   (reads logs/cv_kumo_*_oof.npy and logs/xgb_*_fold*_proba.npy)
import sys
from pathlib import Path

import numpy as np

from msk_data import CLASSES, folds, gene_folds, load_competition, score

LOGS = Path(__file__).parent / "logs"
ALPHAS = np.round(np.arange(0, 0.81, 0.05), 2)
TEMPS = np.round(np.arange(0.3, 1.31, 0.05), 2)


def blend(p, q, alpha):
    return (1 - alpha) * p + alpha * q


def temper(p, t):
    p = np.clip(p, 1e-15, 1) ** t
    return p / p.sum(axis=1, keepdims=True)


def best_hedge(y, p, prior):
    """(log_loss, description) of the best blend and the best temperature."""
    uniform = np.full(len(CLASSES), 1 / len(CLASSES))
    blends = [(score(y, blend(p, q, a))["log_loss"], f"blend {name} alpha={a:.2f}")
              for a in ALPHAS for name, q in [("uniform", uniform), ("prior", prior)]]
    temps = [(score(y, temper(p, t))["log_loss"], f"temperature t={t:.2f}") for t in TEMPS]
    return min(blends), min(temps)


def xgb_oof(tag, splits, n):
    oof = np.zeros((n, len(CLASSES)))
    for k, (_, te) in enumerate(splits):
        oof[te] = np.load(LOGS / f"{tag}_fold{k}_proba.npy")
    return oof


def main():
    variants, _ = load_competition()
    variants = variants[variants["split"] == "train"].reset_index(drop=True)
    y = variants["Class"].astype(int).to_numpy()
    prior = np.bincount(y, minlength=10)[1:] / len(y)
    gene = gene_folds(y, variants["gene_id"])

    names = sys.argv[1:] or sorted(p.name[len("cv_kumo_"):-len("_oof.npy")] for p in LOGS.glob("cv_kumo_*_oof.npy"))
    preds = {f"kumo {n}": np.load(LOGS / f"cv_kumo_{n}_oof.npy") for n in names}
    for tag in sorted({p.name.split("_fold")[0] for p in LOGS.glob("xgb_*_fold0_proba.npy")}):
        preds[f"{tag}"] = xgb_oof(tag, gene if "gene" in tag else folds(y), len(y))

    print(f"{'predictions':34s} {'raw':>7s}  best blend                          best temperature")
    for name, p in preds.items():
        (bl, bd), (tl, td) = best_hedge(y, p, prior)
        print(f"{name:34s} {score(y, p)['log_loss']:7.4f}  {bl:.4f} {bd:28s}  {tl:.4f} {td}")


if __name__ == "__main__":
    main()
