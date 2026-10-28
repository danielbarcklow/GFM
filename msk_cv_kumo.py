# Experiment harness: 5-fold CV on the training rows for a Kumo configuration, with
# fine-tuning redone inside every fold (from the pretrained weights, on that fold's
# training rows only). This is the selection metric for all Kumo experiments;
# stage-1 is never used here.
#
# Usage: python msk_cv_kumo.py --name NAME [--lr 1e-5] [--steps 200] [--context 1024]
#            [--estimators 8] [--seeds 1] [--label-features] [--gene-folds] [--variant-text]
# Appends one line of results to logs/cv_kumo.jsonl and saves the out-of-fold
# probabilities to logs/cv_kumo_NAME_oof.npy (for choosing hedging offline).
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import msk_rfm_finetune as ft
import sdm
from msk_data import CLASSES, SEED, folds, gene_folds, load_competition, score
from msk_features import add_label_features, add_variant_text, article_ids
from msk_rfm_eval import DEVICE, predict

RESULTS = Path(__file__).parent / "logs" / "cv_kumo.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=1, help="fine-tuned copies averaged")
    parser.add_argument("--label-features", action="store_true")
    parser.add_argument("--variant-text", action="store_true", help="add variant-specific text features")
    parser.add_argument("--gene-folds", action="store_true", help="hold out whole genes per fold")
    args = parser.parse_args()
    ft.CONTEXT_SIZE = args.context

    variants, genes = load_competition()
    train_pos = np.flatnonzero(variants["split"] == "train")
    variants = variants.iloc[train_pos].reset_index(drop=True)  # CV uses training rows only
    if args.variant_text:
        variants = add_variant_text(variants)
    y = variants["Class"].astype(int).to_numpy()
    articles = article_ids(variants)

    model = sdm.models.KumoRelational(task="classification", device=DEVICE)
    model.eval()
    pretrained = copy.deepcopy(model.state_dict())

    splits = gene_folds(y, variants["gene_id"]) if args.gene_folds else folds(y)
    oof = np.zeros((len(y), len(CLASSES)))
    rows = []
    start = time.time()
    for k, (tr, te) in enumerate(splits):
        data = variants
        if args.label_features:
            labeled = np.zeros(len(y), bool); labeled[tr] = True
            data = add_label_features(variants, y, labeled, articles)
        ctx = np.zeros(len(y), bool); ctx[tr] = True
        qry = np.zeros(len(y), bool); qry[te] = True

        probas = []
        for s in range(args.seeds):
            model.load_state_dict(pretrained)
            torch.manual_seed(SEED + s)
            if args.steps > 0:
                ft.finetune(model, data, genes, tr, y, args.steps, args.lr, seed=SEED + s)
            probas.append(predict(model, data, genes, ctx, qry, num_estimators=args.estimators))
        oof[np.sort(te)] = np.mean(probas, axis=0)  # predict returns rows in mask order
        rows.append(score(y[np.sort(te)], oof[np.sort(te)]))
        print(f"fold {k}  log_loss={rows[-1]['log_loss']:.4f}  ({time.time() - start:.0f}s)", flush=True)

    mean = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    std = float(np.std([r["log_loss"] for r in rows], ddof=1))
    result = {"name": args.name, **vars(args), **mean, "log_loss_std": std,
              "fold_log_loss": [round(r["log_loss"], 4) for r in rows],
              "minutes": round((time.time() - start) / 60, 1)}
    RESULTS.parent.mkdir(exist_ok=True)
    np.save(RESULTS.parent / f"cv_kumo_{args.name}_oof.npy", oof)
    with RESULTS.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
