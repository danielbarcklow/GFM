# Smoke test from the structured-data-models README (single table, TabICLv2).
import torch
from sklearn.datasets import load_breast_cancer

import sdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

df = load_breast_cancer(as_frame=True).frame
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)

model = sdm.models.TabICLv2(device=device)
probs = model(
    x_context=table[:300].drop_columns("target"),
    y_context=table[:300, "target"],
    x_query=table[300:].drop_columns("target"),
    num_estimators=8,
)
print(probs)
