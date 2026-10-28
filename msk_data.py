# Shared data + evaluation protocol for the MSK "Personalized Medicine: Redefining
# Cancer Treatment" dataset, so the RFM and XGBoost scripts see identical inputs and folds.
#
# Task: classify each (gene, variation) into one of 9 classes from the variant and a
# clinical-literature text. Official metric: multi-class log loss.
#
# Features are built once (label-free) and cached:
#   variants table: one row per variant -> text SVD components, variation-type flags,
#                   protein position, mentions of gene/variation in the text
#   genes table:    one row per gene    -> number of variants, mean text SVD
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

DATA_DIR = Path(__file__).parent / "data" / "msk"
SEED = 0
N_SPLITS = 5
N_REPEATS = 1
CLASSES = list(range(1, 10))
N_TEXT_COMPONENTS = 100
N_GENE_TEXT_COMPONENTS = 20


def _variation_features(variation):
    v = variation.str.strip()
    lower = v.str.lower()
    position = v.str.extract(r"^[A-Z](\d+)", expand=False).astype(float)
    return pd.DataFrame({
        "is_truncating": lower.str.contains("truncating").astype(float),
        "is_deletion": lower.str.contains("deletion|del").astype(float),
        "is_amplification": lower.str.contains("amplification").astype(float),
        "is_fusion": lower.str.contains("fusion").astype(float),
        "is_overexpression": lower.str.contains("overexpression").astype(float),
        "is_insertion": lower.str.contains("ins|dup").astype(float),
        "is_frameshift": lower.str.contains("fs").astype(float),
        "is_splice": lower.str.contains("splice").astype(float),
        "is_point": v.str.fullmatch(r"[A-Z]\d+[A-Z]").astype(float),
        "is_nonsense": v.str.fullmatch(r"[A-Z]\d+\*").astype(float),
        "protein_position": position,
    })


def _read(variants_file, text_file):
    variants = pd.read_csv(DATA_DIR / variants_file)
    text = pd.read_csv(
        DATA_DIR / text_file, sep=r"\|\|", engine="python", skiprows=1, names=["ID", "Text"]
    )
    df = variants.merge(text, on="ID", how="left")
    df["Text"] = df["Text"].fillna("")
    return df


def _solution_labels(solution_file):
    """One-hot Kaggle solution file -> Series of class labels indexed by ID."""
    sol = pd.read_csv(DATA_DIR / solution_file).set_index("ID")
    return sol.to_numpy().argmax(axis=1) + 1, sol.index


def _build_features(df, keep_cols=("Class",)):
    """df: one row per variant with ID, Gene, Variation, Text and `keep_cols`."""
    print("Building text features (one-time, cached)...")
    tfidf = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, min_df=5, max_df=0.9, max_features=50_000
    )
    svd = TruncatedSVD(n_components=N_TEXT_COMPONENTS, random_state=SEED)
    text_svd = svd.fit_transform(tfidf.fit_transform(df["Text"]))
    text_cols = [f"text_{i:03d}" for i in range(N_TEXT_COMPONENTS)]

    def mentions(row, col):
        return len(re.findall(re.escape(row[col]), row["Text"], flags=re.IGNORECASE))

    var = pd.concat(
        [
            df[["ID", "Gene", *keep_cols]],
            _variation_features(df["Variation"]),
            pd.DataFrame({
                "log_text_len": np.log1p(df["Text"].str.len()),
                "log_gene_mentions": np.log1p(df.apply(mentions, axis=1, col="Gene")),
                "log_variation_mentions": np.log1p(df.apply(mentions, axis=1, col="Variation")),
            }),
            pd.DataFrame(text_svd, columns=text_cols),
        ],
        axis=1,
    )

    gene_text = var.groupby("Gene")[text_cols[:N_GENE_TEXT_COMPONENTS]].mean()
    gene_text.columns = [f"gene_{c}" for c in gene_text.columns]
    genes = gene_text.assign(log_num_variants=np.log1p(var["Gene"].value_counts())).reset_index()
    genes.insert(0, "gene_id", np.arange(len(genes)))

    var = var.merge(genes[["Gene", "gene_id"]], on="Gene").sort_values("ID").reset_index(drop=True)
    var = var.drop(columns="Gene").rename(columns={"ID": "variant_id"})
    genes = genes.drop(columns="Gene")
    return var, genes


def load_data():
    """Returns (variants, genes, y). `variants.gene_id` links to `genes.gene_id`."""
    var_path, gene_path = DATA_DIR / "variants_features.parquet", DATA_DIR / "genes_features.parquet"
    if not var_path.exists():
        var, genes = _build_features(_read("training_variants", "training_text"))
        var.to_parquet(var_path)
        genes.to_parquet(gene_path)
    var, genes = pd.read_parquet(var_path), pd.read_parquet(gene_path)
    y = var.pop("Class").to_numpy()
    return var, genes, y


def load_competition():
    """Training rows plus both Kaggle test sets, with features built over all of them.

    Returns (variants, genes) where variants has columns `split` and `Class`:
      split="train"   the 3,321 training rows (Class known)
      split="stage1"  the 368 stage-1 test rows with released labels (development set)
      split="stage2"  all 986 stage-2 test rows; Class is known only for the 125 rows of
                      the private leaderboard solution (NaN otherwise)
    `kaggle_id` is the row's ID within its own Kaggle file (needed for a submission).
    """
    var_path, gene_path = DATA_DIR / "comp_variants.parquet", DATA_DIR / "comp_genes.parquet"
    if not var_path.exists():
        train = _read("training_variants", "training_text").assign(split="train")

        stage1 = _read("test_variants", "test_text")
        labels, ids = _solution_labels("stage1_solution_filtered.csv")
        stage1 = stage1.set_index("ID").loc[ids].reset_index().assign(split="stage1", Class=labels)

        stage2 = _read("stage2_test_variants.csv", "stage2_test_text.csv").assign(split="stage2")
        labels, ids = _solution_labels("stage_2_private_solution.csv")
        stage2["Class"] = stage2["ID"].map(pd.Series(labels, index=ids))

        df = pd.concat([train, stage1, stage2], ignore_index=True)
        df = df.rename(columns={"ID": "kaggle_id"}).rename_axis("ID").reset_index()
        var, genes = _build_features(df, keep_cols=("Class", "split", "kaggle_id"))
        var.to_parquet(var_path)
        genes.to_parquet(gene_path)
    return pd.read_parquet(var_path), pd.read_parquet(gene_path)


def folds(y):
    """Repeated stratified K-fold: (train_idx, test_idx) pairs."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


def gene_folds(y, genes):
    """K-fold with every gene's rows in one fold: tests on genes never seen in training,
    a stand-in for the shift between training and stage-2 rows."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y, groups=genes))


def score(y_true, proba):
    """Metrics from an (n, 9) probability matrix with columns ordered as CLASSES."""
    proba = np.clip(proba, 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        "log_loss": log_loss(y_true, proba, labels=CLASSES),
        "accuracy": accuracy_score(y_true, np.asarray(CLASSES)[proba.argmax(axis=1)]),
        # one-vs-rest AUC averaged over the classes present (a gene-held-out fold can lack one)
        "macro_auc": float(np.mean([roc_auc_score(np.asarray(y_true) == c, proba[:, i])
                                    for i, c in enumerate(CLASSES) if (np.asarray(y_true) == c).any()])),
    }


def report(name, rows):
    """Print per-fold metrics and mean ± std."""
    res = pd.DataFrame(rows)
    res.index.name = "fold"
    print(f"\n{name}: per-fold results")
    print(res.round(4).to_string())
    print(f"\n{name}: mean ± std over {len(res)} folds")
    for col in ["log_loss", "accuracy", "macro_auc"]:
        print(f"  {col:9s} {res[col].mean():.4f} ± {res[col].std():.4f}")
    return res
