# MSK cancer-variant classification (9 classes) with NVIDIA Kumo Relational
# (pretrained, no training). The model sees two linked tables:
#   variants (one row per mutation)  --gene_id-->  genes (one row per gene)
# Training variants are passed as labeled in-context examples at prediction time.
import time

import numpy as np
import torch

import sdm
from sdm import RelatedTables, TableTensor

from msk_data import CLASSES, folds, load_data, report, score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ESTIMATORS = 8  # ensemble members per prediction (package default)

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


def main():
    variants, genes, y = load_data()
    genes_table = to_table(genes, ids={"gene_id"})

    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    print(f"device: {DEVICE}")

    def related(idx):
        return RelatedTables(
            tables={
                "variants": to_table(variants.iloc[idx], ids={"variant_id", "gene_id"}),
                "genes": genes_table,
            },
            relationships=RELATIONSHIPS,
            task_links=TASK_LINKS,
        )

    rows = []
    start = time.time()
    for i, (tr, te) in enumerate(folds(y)):
        task_context = TableTensor.from_pandas(
            df=variants.iloc[tr][["variant_id"]].assign(target=y[tr]).reset_index(drop=True),
            stypes={"variant_id": "id", "target": "categorical"},
            device=DEVICE,
        )
        task_query = to_table(variants.iloc[te][["variant_id"]], ids={"variant_id"})

        out = model(
            x_context=task_context.drop_columns("target"),
            y_context=task_context[:, "target"],
            x_query=task_query,
            related_context_tables=related(tr),
            related_query_tables=related(te),
            num_estimators=NUM_ESTIMATORS,
        )
        # Output columns are named by class label, in no guaranteed order.
        names = list(out.columns[sdm.Stype.numerical])
        probs = out.numerical.cpu().numpy()
        proba = np.stack([probs[:, names.index(str(c))] for c in CLASSES], axis=1)

        rows.append(score(y[te], proba))
        print(f"fold {i}  log_loss={rows[-1]['log_loss']:.4f}  ({time.time() - start:.0f}s)")

    report("Kumo Relational", rows)


if __name__ == "__main__":
    main()
