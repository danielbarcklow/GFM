# MSK cancer-variant classification (9 classes) with XGBoost, tuned by nested CV.
# XGBoost gets the same information as msk_rfm.py, with the genes table joined
# onto each variant as extra columns.
# Outer loop: the same folds as msk_rfm.py, used only for scoring.
# Inner loop: on each outer training set, Optuna searches hyperparameters,
# scoring each candidate by 5-fold CV log loss with early stopping. The best
# configuration is then refit on the full outer training set.
#
# Usage:
#   python msk_xgb.py [n_trials]                 all outer folds in sequence (default 50 trials)
#   python msk_xgb.py 100 --fold 2 --device cpu --jobs 6
#                                                one outer fold; saves logs/xgb_fold2.json
#   python msk_xgb.py --collect [--tag xgb]      report the saved per-fold results
#   add --label-features --tag xgb_label          to add label-sharing features (computed per fold)
#   add --comp-features --tag xgb_comp            competition feature build, no label features
#   add --gene-folds                              outer folds hold out whole genes
#   add --variant-text                            variant-specific text features (competition build)
# Per-fold runs also save the fold's test probabilities, logs/TAG_foldK_proba.npy.
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import optuna
import xgboost as xgb

import numpy as np

from msk_data import CLASSES, SEED, folds, gene_folds, load_competition, load_data, report, score
from msk_features import add_label_features, add_variant_text, article_ids

N_TRIALS = 50
INNER_FOLDS = 5
MAX_ROUNDS = 3000
EARLY_STOPPING = 50
RESULTS_DIR = Path(__file__).parent / "logs"

# The data is small, so one XGBoost fit gains little from many threads. On CPU, run
# many single-threaded trials in parallel instead; on a GPU, share it among a few trials.
DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
PARALLEL_TRIALS = 4 if DEVICE == "cuda" else os.cpu_count()

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial):
    return {
        "objective": "multi:softprob",
        "num_class": len(CLASSES),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": DEVICE,
        "seed": SEED,
        "nthread": 1 if DEVICE == "cpu" else -1,
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }


def tune(X, y):
    """Return (params, num_boost_round) minimizing inner-CV log loss."""
    dtrain = xgb.DMatrix(X, label=y)

    def objective(trial):
        params = suggest_params(trial)
        cv = xgb.cv(
            params,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            nfold=INNER_FOLDS,
            stratified=True,
            early_stopping_rounds=EARLY_STOPPING,
            seed=SEED,
        )
        trial.set_user_attr("num_boost_round", len(cv))
        return cv["test-mlogloss-mean"].iloc[-1]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=PARALLEL_TRIALS)
    best = study.best_trial
    return suggest_params(optuna.trial.FixedTrial(best.params)), best.user_attrs["num_boost_round"]


def run_fold(i, tr, te, X, y):
    """Tune on the outer training rows, refit, and score the outer test rows."""
    y0 = y - 1  # XGBoost wants labels 0..8
    start = time.time()
    params, num_rounds = tune(X.iloc[tr], y0[tr])
    booster = xgb.train(params, xgb.DMatrix(X.iloc[tr], label=y0[tr]), num_boost_round=num_rounds)
    proba = booster.predict(xgb.DMatrix(X.iloc[te]))
    result = score(y[te], proba)
    print(
        f"fold {i}  log_loss={result['log_loss']:.4f}  rounds={num_rounds}  "
        f"depth={params['max_depth']}  eta={params['eta']:.3f}  ({time.time() - start:.0f}s)",
        flush=True,
    )
    return result, proba


def main():
    global N_TRIALS, DEVICE, PARALLEL_TRIALS
    parser = argparse.ArgumentParser()
    parser.add_argument("n_trials", nargs="?", type=int, default=N_TRIALS)
    parser.add_argument("--fold", type=int, help="run only this outer fold and save its result")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available")
    parser.add_argument("--jobs", type=int, help="trials run in parallel")
    parser.add_argument("--collect", action="store_true", help="report saved per-fold results")
    parser.add_argument("--label-features", action="store_true",
                        help="use the competition feature build plus label-sharing features")
    parser.add_argument("--comp-features", action="store_true",
                        help="use the competition feature build (TF-IDF fit on train+stage1+stage2)")
    parser.add_argument("--variant-text", action="store_true", help="add variant-specific text features")
    parser.add_argument("--gene-folds", action="store_true", help="outer folds hold out whole genes")
    parser.add_argument("--tag", default="xgb", help="prefix for saved per-fold results")
    args = parser.parse_args()
    N_TRIALS = args.n_trials
    DEVICE = args.device or DEVICE
    PARALLEL_TRIALS = args.jobs or (4 if DEVICE == "cuda" else os.cpu_count())

    if args.collect:
        paths = sorted(RESULTS_DIR.glob(f"{args.tag}_fold*.json"))
        report(f"XGBoost (nested-CV tuned, {len(paths)} folds)", [json.loads(p.read_text()) for p in paths])
        return

    if args.label_features or args.comp_features or args.variant_text:
        variants, genes = load_competition()
        variants = variants[variants["split"] == "train"].reset_index(drop=True)
        y = variants.pop("Class").astype(int).to_numpy()
        articles = article_ids(variants)
        if args.variant_text:
            variants = add_variant_text(variants)
        variants = variants.drop(columns=["split", "kaggle_id"])
    else:
        variants, genes, y = load_data()
    splits = gene_folds(y, variants["gene_id"]) if args.gene_folds else folds(y)

    def features(tr):
        data = variants
        if args.label_features:
            labeled = np.zeros(len(y), bool)
            labeled[tr] = True
            data = add_label_features(variants, y, labeled, articles)
        return data.merge(genes, on="gene_id", how="left").drop(columns=["variant_id", "gene_id"])

    if args.fold is not None:
        tr, te = splits[args.fold]
        RESULTS_DIR.mkdir(exist_ok=True)
        result, proba = run_fold(args.fold, tr, te, features(tr), y)
        (RESULTS_DIR / f"{args.tag}_fold{args.fold}.json").write_text(json.dumps(result))
        np.save(RESULTS_DIR / f"{args.tag}_fold{args.fold}_proba.npy", proba)
        return

    rows = [run_fold(i, tr, te, features(tr), y)[0] for i, (tr, te) in enumerate(splits)]
    report(f"XGBoost (nested-CV tuned, {N_TRIALS} trials)", rows)


if __name__ == "__main__":
    main()
