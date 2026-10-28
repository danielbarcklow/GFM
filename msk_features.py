# Extra MSK features that use labels, computed so that no row ever sees its own label.
#
# Label-sharing features: for each variant, the class mix of *other* labeled variants
# with the same gene, and with the same article text. Labeled rows get out-of-fold
# values (statistics from labeled rows in the other folds); unlabeled rows (test) use
# all labeled rows. Smoothed toward the overall class mix so rare genes/articles don't
# produce extreme values.
#
# Not leave-one-out: removing a row's own label makes its own class look *rarer* in
# its group than it does for its neighbours, an inverted signal that models learn
# perfectly on training rows and that fails on test rows.
import hashlib
import re

import numpy as np
import pandas as pd

from msk_data import CLASSES, DATA_DIR, _read

SMOOTHING = 1.0  # pseudo-count weight on the overall class mix
ENCODING_FOLDS = 5


def article_ids(variants):
    """An id per variant identifying its article text (identical texts share an id)."""
    path = DATA_DIR / "comp_article_ids.parquet"
    if not path.exists():
        files = {
            "train": ("training_variants", "training_text"),
            "stage1": ("test_variants", "test_text"),
            "stage2": ("stage2_test_variants.csv", "stage2_test_text.csv"),
        }
        parts = []
        for split, (vf, tf) in files.items():
            df = _read(vf, tf)
            df["article"] = df["Text"].map(lambda t: hashlib.md5(t.encode()).hexdigest())
            parts.append(df[["ID", "article"]].assign(split=split).rename(columns={"ID": "kaggle_id"}))
        pd.concat(parts).to_parquet(path)
    ids = pd.read_parquet(path)
    merged = variants[["split", "kaggle_id"]].merge(ids, on=["split", "kaggle_id"], how="left")
    return pd.factorize(merged["article"])[0]


def label_sharing_features(groups, y, labeled):
    """Class-mix features per group.

    groups:  dict name -> integer group id per row (e.g. gene_id, article id)
    y:       class label per row (only used where `labeled`)
    labeled: boolean mask of rows whose labels may be used
    """
    n = len(y)
    prior = np.bincount(y[labeled], minlength=10)[1:].astype(float)
    prior /= prior.sum()
    onehot = np.zeros((n, len(CLASSES)))
    onehot[labeled, y[labeled] - 1] = 1.0

    fold = np.full(n, -1)  # -1 = unlabeled: sees every labeled row
    fold[labeled] = np.random.default_rng(0).permutation(labeled.sum()) % ENCODING_FOLDS

    cols = {}
    for name, g in groups.items():
        g = np.asarray(g)
        sums = np.zeros((n, len(CLASSES)))
        for k in range(-1, ENCODING_FOLDS):
            rows = fold == k
            seen = onehot * (labeled & (fold != k))[:, None]
            sums[rows] = pd.DataFrame(seen).groupby(g).transform("sum").to_numpy()[rows]
        counts = sums.sum(axis=1)
        mix = (sums + SMOOTHING * prior) / (counts + SMOOTHING)[:, None]
        for i, c in enumerate(CLASSES):
            cols[f"{name}_class{c}"] = mix[:, i]
        cols[f"{name}_log_n"] = np.log1p(counts)
    return pd.DataFrame(cols)


def add_label_features(variants, y, labeled, articles):
    """variants with label-sharing columns appended (same row order)."""
    feats = label_sharing_features(
        {"gene": variants["gene_id"].to_numpy(), "article": articles}, y, labeled
    )
    return pd.concat([variants.reset_index(drop=True), feats], axis=1)


# Variant-specific text: many articles discuss dozens of mutations, so a summary of the
# whole article blurs them together. Keep only the sentences that mention this row's
# mutation (exact variant, e.g. "V600E", or the same residue, e.g. "V600"); for
# non-point variants ("Truncating Mutations", "Amplification", ...) the sentences that
# mention the gene together with the variant's keyword. Label-free, cached.
N_VARIANT_TEXT_COMPONENTS = 50
KEYWORDS = {"truncat": "truncat", "delet": "delet", "amplif": "amplif", "fusion": "fusion",
            "overexpress": "overexpress", "splice": "splice", "promoter": "promoter"}
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _variant_sentences(gene, variation, text):
    sentences = SENTENCE.split(text)
    point = re.fullmatch(r"([A-Z])(\d+)([A-Z*]?)", variation.strip())
    if point:
        residue = re.compile(rf"\b{point[1]}{point[2]}(?:[A-Z*]|\b)")
        hits = [s for s in sentences if residue.search(s)]
    else:
        lower = variation.lower()
        keys = [k for k in KEYWORDS if k in lower]
        exact = re.compile(re.escape(variation.strip()), re.IGNORECASE)
        hits = [s for s in sentences if exact.search(s)
                or (gene in s and any(k in s.lower() for k in keys))]
    return " ".join(hits)


def variant_text_features(variants):
    """Columns vtext_000.. and log_variant_sentences for `variants` (any splits of the
    competition build), aligned to its row order."""
    path = DATA_DIR / "comp_variant_text.parquet"
    if not path.exists():
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        files = {
            "train": ("training_variants", "training_text"),
            "stage1": ("test_variants", "test_text"),
            "stage2": ("stage2_test_variants.csv", "stage2_test_text.csv"),
        }
        df = pd.concat([_read(vf, tf).assign(split=s) for s, (vf, tf) in files.items()], ignore_index=True)
        print("Building variant-specific text features (one-time, cached)...", flush=True)
        snippets = [_variant_sentences(g, v, t) for g, v, t in zip(df["Gene"], df["Variation"], df["Text"])]
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2),
                                min_df=3, max_df=0.9, max_features=50_000)
        svd = TruncatedSVD(n_components=N_VARIANT_TEXT_COMPONENTS, random_state=0)
        comps = svd.fit_transform(tfidf.fit_transform(snippets))
        out = pd.DataFrame(comps, columns=[f"vtext_{i:03d}" for i in range(comps.shape[1])])
        out["log_variant_sentences"] = np.log1p([s.count(". ") + bool(s) for s in snippets])
        out.insert(0, "kaggle_id", df["ID"].to_numpy())
        out.insert(0, "split", df["split"].to_numpy())
        out.to_parquet(path)
        print(f"  {np.mean([bool(s) for s in snippets]):.1%} of rows have variant sentences", flush=True)
    feats = pd.read_parquet(path)
    merged = variants[["split", "kaggle_id"]].merge(feats, on=["split", "kaggle_id"], how="left")
    return merged.drop(columns=["split", "kaggle_id"])


def add_variant_text(variants):
    """variants with the variant-specific text columns appended (same row order)."""
    return pd.concat([variants.reset_index(drop=True), variant_text_features(variants)], axis=1)
