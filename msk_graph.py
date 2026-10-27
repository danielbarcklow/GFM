# Per-example subgraphs for Kumo Relational on the MSK data.
#
# KumoRelational expects every task row to come with its own, non-overlapping
# neighborhood (it says so in sdm/models/kumo/relational/task.py). We build that
# directly, the way sdm's RelationalSampler would, without needing pyg-lib:
#
#   hop 1: the seed variant  -> a copy of its gene row
#   hop 2: that gene         -> up to `fanout` other variants of the same gene
#
# Every copied row carries an "__example__" id (the seed's position), and all
# keys are composite (key, __example__), so neighborhoods never overlap.
import numpy as np
import pandas as pd
import torch

import sdm
from sdm import RelatedTables, TableTensor

EXAMPLE_ID = "__example__"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_table(df, ids):
    stypes = {c: "id" if c in ids else "numerical" for c in df.columns}
    return TableTensor.from_pandas(df=df.reset_index(drop=True), stypes=stypes, device=DEVICE)


def build_subgraphs(seeds, variants, genes, pool, num_hops, fanout, seed=0):
    """RelatedTables holding one disjoint neighborhood per seed variant.

    seeds:    variant_ids to predict for (in task-table order)
    variants: variant feature table indexed by variant_id (features only, no labels)
    genes:    gene feature table with gene_id
    pool:     variant_ids that may appear as hop-2 neighbors (features only)
    """
    rng = np.random.default_rng(seed)
    seed_rows = variants.loc[seeds].reset_index()
    seed_rows[EXAMPLE_ID] = np.arange(len(seeds))
    parts = [seed_rows]

    if num_hops >= 2 and fanout > 0:
        pool_rows = variants.loc[variants.index.intersection(pool)].reset_index()
        by_gene = pool_rows.groupby("gene_id").indices
        for ex, (vid, gid) in enumerate(zip(seed_rows["variant_id"], seed_rows["gene_id"])):
            idx = by_gene.get(gid, np.array([], dtype=int))
            idx = idx[pool_rows["variant_id"].to_numpy()[idx] != vid]
            if len(idx) > fanout:
                idx = rng.choice(idx, size=fanout, replace=False)
            if len(idx):
                parts.append(pool_rows.iloc[idx].assign(**{EXAMPLE_ID: ex}))
    var_table = pd.concat(parts, ignore_index=True)

    gene_table = seed_rows[["gene_id", EXAMPLE_ID]].merge(genes, on="gene_id", how="left")

    tables = {"variants": _to_table(var_table, ids={"variant_id", "gene_id", EXAMPLE_ID})}
    relationships = []
    if num_hops >= 1:
        tables["genes"] = _to_table(gene_table, ids={"gene_id", EXAMPLE_ID})
        relationships.append({
            "left_table": "variants",
            "left_columns": ["gene_id", EXAMPLE_ID],
            "right_table": "genes",
            "right_columns": ["gene_id", EXAMPLE_ID],
        })
    return RelatedTables(
        tables=tables,
        relationships=relationships,
        task_links=[{
            "task_columns": ["variant_id", EXAMPLE_ID],
            "table": "variants",
            "table_columns": ["variant_id", EXAMPLE_ID],
        }],
    )


def task_table(seeds, labels=None):
    df = pd.DataFrame({"variant_id": seeds, EXAMPLE_ID: np.arange(len(seeds))})
    stypes = {"variant_id": "id", EXAMPLE_ID: "id"}
    if labels is not None:
        df["target"] = np.asarray(labels, dtype=int)
        stypes["target"] = "categorical"
    return TableTensor.from_pandas(df=df, stypes=stypes, device=DEVICE)


def predict(model, variants, genes, context, y_context, query, pool,
            num_hops=2, fanout=10, num_estimators=8, seed=0):
    """Class probabilities (columns ordered 1..9) for `query` variant_ids."""
    ctx = task_table(context, y_context)
    out = model(
        x_context=ctx.drop_columns("target"),
        y_context=ctx[:, "target"],
        x_query=task_table(query),
        related_context_tables=build_subgraphs(context, variants, genes, pool, num_hops, fanout, seed),
        related_query_tables=build_subgraphs(query, variants, genes, pool, num_hops, fanout, seed + 1),
        num_hops=num_hops,
        num_estimators=num_estimators,
    )
    names = list(out.columns[sdm.Stype.numerical])
    probs = out.numerical.cpu().numpy()
    return np.stack([probs[:, names.index(str(c))] for c in range(1, 10)], axis=1)
