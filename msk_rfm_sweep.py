# Step 3: tune Kumo Relational's own settings, selecting on training data only.
# Each configuration is scored by the same 5-fold CV on the training rows that
# msk_rfm.py / msk_xgb.py use. The stage-1 set is not touched here.
#
# Usage: python msk_rfm_sweep.py
import itertools
import time

import pandas as pd

import sdm

from msk_data import folds, load_data, score
from msk_graph import DEVICE, predict

GRID = {
    "num_hops": [1, 2],         # 1: variant -> gene;  2: also the gene's other variants
    "fanout": [5, 20],          # other variants sampled per gene (hop 2 only)
    "num_estimators": [8, 32],
}


def main():
    variants, genes, y = load_data()
    variants = variants.set_index("variant_id", drop=False).rename_axis(None)
    feats = variants.drop(columns="variant_id").rename_axis("variant_id")
    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    splits = folds(y)

    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    configs = [c for c in configs if c["num_hops"] == 2 or c["fanout"] == GRID["fanout"][0]]

    rows = []
    for cfg in configs:
        start = time.time()
        scores = []
        for tr, te in splits:
            ids = variants["variant_id"].to_numpy()
            proba = predict(model, feats, genes, ids[tr], y[tr], ids[te], pool=ids[tr], **cfg)
            scores.append(score(y[te], proba))
        res = pd.DataFrame(scores).mean().to_dict()
        rows.append({**cfg, **res, "seconds": round(time.time() - start)})
        print(rows[-1], flush=True)

    out = pd.DataFrame(rows).sort_values("log_loss")
    print("\n5-fold CV on training rows, sorted by log loss (lower is better):")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
