# Fine-tune Kumo Relational on the MSK training rows (episodic in-context training).
#
# Each step samples a random "episode" from the fine-tuning rows: some rows act as
# labeled context, others as queries, and we minimise the queries' log loss. This
# matches how the model is used at prediction time.
#
# Protocol (stage-1 is never used for any choice):
#   1. Split the training rows 80/20 (stratified). Fine-tune on the 80%, scoring the
#      held-out 20% every EVAL_EVERY steps (context = the 80%).
#   2. Take the best step count, re-fine-tune from the pretrained weights on all
#      training rows for that many steps.
#   3. Score once on the 368 labeled stage-1 rows (context = all training rows).
#
# Usage: python msk_rfm_finetune.py [--lr 1e-5] [--max-steps 600] [--seed 0] [--out rfm_finetuned]
import argparse
import copy
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import sdm

from msk_data import CLASSES, DATA_DIR, SEED, load_competition, score
from msk_rfm_eval import DEVICE, predict, run_model

CONTEXT_SIZE = 1024
QUERY_SIZE = 256
EVAL_EVERY = 50


def sample_episode(rows, y, rng):
    """Disjoint (context, query) row positions; context holds every class present."""
    perm = rng.permutation(rows)
    context, query = list(perm[:CONTEXT_SIZE]), perm[CONTEXT_SIZE:CONTEXT_SIZE + QUERY_SIZE]
    for c in CLASSES:  # make sure every class appears in the context
        if not np.any(y[context] == c):
            pool = [r for r in rows if y[r] == c and r not in set(query)]
            if pool:
                context.append(rng.choice(pool))
    return np.array(context), np.array(query)


def finetune(model, variants, genes, rows, y, steps, lr, holdout=None, seed=SEED):
    """Fine-tune in place. If `holdout` is given, return [(step, holdout log loss), ...]."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []

    def evaluate(step):
        model.eval()
        mask = np.zeros(len(variants), bool)
        ctx = mask.copy(); ctx[rows] = True
        qry = mask.copy(); qry[holdout] = True
        s = score(y[holdout], predict(model, variants, genes, ctx, qry))
        history.append((step, s["log_loss"]))
        print(f"  step {step:4d}  holdout log_loss={s['log_loss']:.4f}  accuracy={s['accuracy']:.4f}", flush=True)

    if holdout is not None:
        evaluate(0)
    start = time.time()
    for step in range(1, steps + 1):
        model.train()
        context, query = sample_episode(rows, y, rng)
        mask = np.zeros(len(variants), bool)
        ctx = mask.copy(); ctx[context] = True
        qry = mask.copy(); qry[query] = True
        # Row order inside run_model follows the boolean masks, i.e. sorted positions.
        names, probs = run_model(model, variants, genes, ctx, qry, num_estimators=1)
        y_query = y[np.sort(query)]
        col = {int(n): i for i, n in enumerate(names)}
        keep = np.array([c in col for c in y_query])
        idx = torch.tensor([col[c] for c in y_query[keep]], device=probs.device)
        p_true = probs[torch.as_tensor(keep, device=probs.device)].gather(1, idx[:, None]).squeeze(1)
        loss = -torch.log(p_true.clamp_min(1e-7)).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            print(f"  step {step:4d}  train loss={loss.item():.4f}  ({time.time() - start:.0f}s)", flush=True)
        if holdout is not None and step % EVAL_EVERY == 0:
            evaluate(step)
    model.eval()
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=SEED, help="episode sampling seed")
    parser.add_argument("--out", default="rfm_finetuned", help="name for saved weights/predictions")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    variants, genes = load_competition()
    y = variants["Class"].fillna(0).astype(int).to_numpy()
    train_rows = np.flatnonzero(variants["split"] == "train")
    dev = (variants["split"] == "stage1").to_numpy()
    ctx_all = (variants["split"] == "train").to_numpy()

    pretrained = sdm.models.KumoRelational(task="classification", device=DEVICE)
    pretrained.eval()
    pretrained_state = copy.deepcopy(pretrained.state_dict())

    # 1. Choose the number of steps on an 80/20 split of the training rows.
    ft_rows, holdout = train_test_split(train_rows, test_size=0.2, stratify=y[train_rows], random_state=SEED)
    holdout = np.sort(holdout)  # predict() returns query rows in sorted position order
    print(f"Phase 1: fine-tune on {len(ft_rows)} rows, check on {len(holdout)} held-out training rows (lr={args.lr})")
    history = finetune(pretrained, variants, genes, ft_rows, y, args.max_steps, args.lr, holdout=holdout, seed=args.seed)
    best_step, best_loss = min(history, key=lambda h: h[1])
    print(f"Best: step {best_step} (holdout log_loss {best_loss:.4f}; step 0 = pretrained {history[0][1]:.4f})")

    # 2. Re-fine-tune from the pretrained weights on all training rows.
    pretrained.load_state_dict(pretrained_state)
    if best_step > 0:
        print(f"\nPhase 2: fine-tune on all {len(train_rows)} training rows for {best_step} steps")
        finetune(pretrained, variants, genes, train_rows, y, best_step, args.lr, seed=args.seed)
    torch.save(pretrained.state_dict(), DATA_DIR / f"{args.out}.pt")

    # 3. Score once on stage-1.
    proba = predict(pretrained, variants, genes, ctx_all, dev)
    np.save(DATA_DIR / f"{args.out}_stage1_proba.npy", proba)
    s = score(y[dev], proba)
    print(f"\nFine-tuned Kumo Relational on stage-1 (368 rows): log_loss={s['log_loss']:.4f}  "
          f"accuracy={s['accuracy']:.4f}  macro_auc={s['macro_auc']:.4f}")
    print("Reference: pretrained Kumo 0.731, tuned XGBoost 0.731")


if __name__ == "__main__":
    main()
