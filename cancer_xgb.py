# Breast cancer classification with XGBoost, tuned by nested cross-validation.
# Outer loop: the same folds as cancer_rfm.py, used only for scoring.
# Inner loop: on each outer training set, Optuna searches hyperparameters,
# scoring each candidate by 5-fold CV log loss with early stopping. The best
# configuration is then refit on the full outer training set.
import time

import numpy as np
import optuna
import xgboost as xgb

from cancer_data import SEED, folds, load_data, report, score

N_TRIALS = 100
INNER_FOLDS = 5
MAX_ROUNDS = 2000
EARLY_STOPPING = 50

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial):
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": -1,
        "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
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
        return cv["test-logloss-mean"].iloc[-1]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    return suggest_params(optuna.trial.FixedTrial(best.params)), best.user_attrs["num_boost_round"]


def main():
    X, y = load_data()

    rows = []
    start = time.time()
    for i, (tr, te) in enumerate(folds(y)):
        params, num_rounds = tune(X.iloc[tr], y[tr])
        booster = xgb.train(params, xgb.DMatrix(X.iloc[tr], label=y[tr]), num_boost_round=num_rounds)
        p_pos = booster.predict(xgb.DMatrix(X.iloc[te]))

        rows.append(score(y[te], p_pos))
        print(
            f"fold {i:2d}  auc={rows[-1]['roc_auc']:.4f}  rounds={num_rounds}  "
            f"depth={params['max_depth']}  eta={params['eta']:.3f}  ({time.time() - start:.0f}s)"
        )

    report("XGBoost (nested-CV tuned)", rows)


if __name__ == "__main__":
    main()
