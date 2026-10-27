# Shared data + evaluation protocol so the RFM and XGBoost scripts see identical folds.
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 0
N_SPLITS = 5
N_REPEATS = 3


def load_data():
    """Wisconsin Diagnostic Breast Cancer: 569 patients, 30 numeric features,
    target 0 = malignant, 1 = benign."""
    df = load_breast_cancer(as_frame=True).frame
    X = df.drop(columns="target")
    y = df["target"].to_numpy()
    return X, y


def folds(y):
    """Repeated stratified K-fold: (train_idx, test_idx) pairs."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


def score(y_true, p_pos):
    """Metrics from the predicted probability of class 1."""
    return {
        "roc_auc": roc_auc_score(y_true, p_pos),
        "accuracy": accuracy_score(y_true, (p_pos >= 0.5).astype(int)),
        "log_loss": log_loss(y_true, p_pos, labels=[0, 1]),
    }


def report(name, rows):
    """Print per-fold metrics and mean ± std."""
    res = pd.DataFrame(rows)
    res.index.name = "fold"
    print(f"\n{name}: per-fold results")
    print(res.round(4).to_string())
    print(f"\n{name}: mean ± std over {len(res)} folds")
    for col in ["roc_auc", "accuracy", "log_loss"]:
        print(f"  {col:9s} {res[col].mean():.4f} ± {res[col].std():.4f}")
    return res
