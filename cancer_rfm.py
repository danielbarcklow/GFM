# Breast cancer classification with NVIDIA Kumo Relational (pretrained, no training).
# The model learns in context: training rows are passed as labeled examples at
# prediction time, so there is nothing to fit or tune.
import time

import numpy as np
import torch

import sdm
from sdm import RelatedTables, TableTensor

from cancer_data import folds, load_data, report, score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ESTIMATORS = 8  # ensemble members per prediction (package default)

# Kumo Relational expects a task table linked to one or more related tables.
# This dataset is a single table, so the task table holds just patient_id
# (and the label), and a "patients" table holds the features.
TASK_LINKS = [{"task_columns": "patient_id", "table": "patients", "table_columns": "patient_id"}]


def to_table(df, stypes):
    return TableTensor.from_pandas(df=df.reset_index(drop=True), stypes=stypes, device=DEVICE)


def main():
    X, y = load_data()
    X = X.assign(patient_id=np.arange(len(X)))
    patient_stypes = {c: "numerical" for c in X.columns}
    patient_stypes["patient_id"] = "id"

    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    print(f"device: {DEVICE}")

    rows = []
    start = time.time()
    for i, (tr, te) in enumerate(folds(y)):
        task_context = to_table(
            X.iloc[tr][["patient_id"]].assign(target=y[tr]),
            {"patient_id": "id", "target": "categorical"},
        )
        task_query = to_table(X.iloc[te][["patient_id"]], {"patient_id": "id"})
        related = lambda idx: RelatedTables(
            tables={"patients": to_table(X.iloc[idx], patient_stypes)},
            relationships=[],
            task_links=TASK_LINKS,
        )

        out = model(
            x_context=task_context.drop_columns("target"),
            y_context=task_context[:, "target"],
            x_query=task_query,
            related_context_tables=related(tr),
            related_query_tables=related(te),
            num_hops=1,
            num_estimators=NUM_ESTIMATORS,
        )
        # Output columns are named by class label, in no guaranteed order.
        classes = out.columns[sdm.Stype.numerical]
        p_pos = out.numerical[:, classes.index("1")].cpu().numpy()

        rows.append(score(y[te], p_pos))
        print(f"fold {i:2d}  auc={rows[-1]['roc_auc']:.4f}  ({time.time() - start:.0f}s)")

    report("Kumo Relational", rows)


if __name__ == "__main__":
    main()
