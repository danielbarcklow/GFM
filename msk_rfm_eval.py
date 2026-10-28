# Kumo Relational on the MSK competition protocol (no Kaggle submission).
#   context (labeled examples): training rows
#   development set:            368 labeled stage-1 test rows -> all choices are made here
#
# Step 2, "hedging": log loss punishes confident mistakes, so we blend the model's
# probabilities with the training class frequencies:
#     p_hedged = (1 - alpha) * p_model + alpha * p_prior
# and pick alpha on the development set.
import numpy as np
import pandas as pd
import torch

import sdm
from sdm import RelatedTables, TableTensor

from msk_data import CLASSES, DATA_DIR, load_competition, score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ESTIMATORS = 8
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)

RELATIONSHIPS = [{
    "left_table": "variants",
    "left_columns": "gene_id",
    "right_table": "genes",
    "right_columns": "gene_id",
}]
TASK_LINKS = [{"task_columns": "variant_id", "table": "variants", "table_columns": "variant_id"}]


def to_table(df, ids):
    stypes = {c: "id" if c in ids else "numerical" for c in df.columns}
    return TableTensor.from_pandas(df=df.reset_index(drop=True), stypes=stypes, device=DEVICE)


def run_model(model, variants, genes, context, query, num_estimators=NUM_ESTIMATORS):
    """Raw model output for boolean row masks: (class names, probability tensor).

    Query rows come out in the order of `query`'s True positions. In training mode
    the probabilities carry gradients.
    """
    features = variants.drop(columns=["Class", "split", "kaggle_id"])
    genes_table = to_table(genes, ids={"gene_id"})

    def related(rows):
        return RelatedTables(
            tables={"variants": to_table(features[rows], ids={"variant_id", "gene_id"}), "genes": genes_table},
            relationships=RELATIONSHIPS,
            task_links=TASK_LINKS,
        )

    task_context = TableTensor.from_pandas(
        df=variants.loc[context, ["variant_id"]].assign(target=variants.loc[context, "Class"].astype(int)).reset_index(drop=True),
        stypes={"variant_id": "id", "target": "categorical"},
        device=DEVICE,
    )
    out = model(
        x_context=task_context.drop_columns("target"),
        y_context=task_context[:, "target"],
        x_query=to_table(variants.loc[query, ["variant_id"]], ids={"variant_id"}),
        related_context_tables=related(context),
        related_query_tables=related(query),
        num_estimators=num_estimators,
    )
    return list(out.columns[sdm.Stype.numerical]), out.numerical


def predict(model, variants, genes, context, query):
    """Class probabilities (columns ordered as CLASSES) for `query` rows given labeled `context` rows."""
    names, probs = run_model(model, variants, genes, context, query)
    probs = probs.detach().cpu().numpy()
    return np.stack([probs[:, names.index(str(c))] for c in CLASSES], axis=1)


def main():
    variants, genes = load_competition()
    model = sdm.models.KumoRelational(task="classification", device=DEVICE)

    context = (variants["split"] == "train").to_numpy()
    dev = (variants["split"] == "stage1").to_numpy()
    y_dev = variants.loc[dev, "Class"].astype(int).to_numpy()

    proba = predict(model, variants, genes, context, dev)
    np.save(DATA_DIR / "rfm_stage1_proba.npy", proba)

    prior = variants.loc[context, "Class"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    uniform = np.full(len(CLASSES), 1 / len(CLASSES))

    rows = []
    for alpha in ALPHAS:
        for name, p in [("train_freq", prior), ("uniform", uniform)]:
            rows.append({"prior": name, "alpha": alpha, **score(y_dev, (1 - alpha) * proba + alpha * p)})
    res = pd.DataFrame(rows)

    print("\nReference points on the development set (stage-1, 368 rows):")
    print(f"  all 1/9 (uniform)      log_loss={score(y_dev, np.tile(uniform, (dev.sum(), 1)))['log_loss']:.4f}")
    print(f"  training class freqs   log_loss={score(y_dev, np.tile(prior, (dev.sum(), 1)))['log_loss']:.4f}")
    print(f"  Kumo Relational, raw   log_loss={res.loc[res.alpha == 0, 'log_loss'].iloc[0]:.4f}")

    print("\nHedging sweep (log loss; lower is better):")
    print(res.pivot(index="alpha", columns="prior", values="log_loss").round(4).to_string())
    best = res.loc[res["log_loss"].idxmin()]
    print(f"\nBest: prior={best['prior']}  alpha={best['alpha']}  log_loss={best['log_loss']:.4f}  "
          f"accuracy={best['accuracy']:.4f}  macro_auc={best['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
