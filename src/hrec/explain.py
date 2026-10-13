"""Turn TreeSHAP contributions into per-recommendation reasons people can read.

SHAP values are additive, so summing them over a feature group ("everything SASRec
said", "everything about the item's popularity") is still a faithful attribution.

Raw attributions make every recommendation look the same (the sequence model and the
graph model dominate everyone's scores). So reasons are *contrastive*: for each item
we pick the groups that push it up more than they push up the user's other top-20
items, i.e. "why this one rather than the rest of your list". Then we attach a
concrete anchor, such as the item in your history the similarity signal leans on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_LABELS = {
    "sasrec": "sequence model (SASRec)",
    "lightgcn": "graph neighbours (LightGCN)",
    "ease": "co-occurrence (EASE)",
    "itemknn": "similar items (ItemKNN)",
    "content": "content match",
    "pop": "recent popularity",
    "agreement": "retriever agreement",
    "item": "item quality/popularity",
    "user": "user context",
    "taste": "taste match",
}


def group_contributions(contrib: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in contrib.columns if c != "bias"]
    groups = sorted({c.split("__")[0] for c in cols})
    return pd.DataFrame({g: contrib[[c for c in cols if c.startswith(g + "__")]].sum(axis=1)
                         for g in groups}, index=contrib.index)


def reason_for(group: str, user: int, item: int, ctx: dict) -> str:
    meta, seq = ctx["item_meta"], ctx["sequences"][user]
    title = lambda i: meta.at[i, "title"]
    if group == "itemknn" and len(seq):
        S = ctx["itemknn_S"]
        anchor = seq[np.argmax(S[seq, item])]
        return f"similar to {title(anchor)!s}, which you watched"
    if group == "sasrec" and len(seq):
        recent = seq[-10:]
        anchor = recent[np.argmax(ctx["itemknn_S"][recent, item])]  # co-occurrence anchor
        return f"follows on from what you've watched lately, especially {title(anchor)}"
    if group == "lightgcn":
        return "people whose overall taste resembles yours watched it"
    if group == "ease":
        return "often watched alongside things in your history"
    if group in ("content", "taste"):
        g = meta.at[item, "genres"]
        recent = {x for i in seq[-10:] for x in meta.at[i, "genres"]}
        shared = [x for x in g if x in recent][:2]
        return f"matches genres you've been watching ({', '.join(shared)})" if shared else "close to your content profile"
    if group == "pop":
        return "trending in the last few weeks"
    if group == "item":
        return "widely watched and well rated"
    if group == "agreement":
        return "several independent models ranked it highly"
    if group == "user":
        return "people with a similar profile tend to pick it"
    return GROUP_LABELS.get(group, group)


def explain(rows: pd.DataFrame, gcontrib: pd.DataFrame, ctx: dict, peers: pd.DataFrame | None = None,
            top: int = 2) -> list[dict]:
    """rows: candidates to explain; peers: the user's wider top list (defaults to rows)."""
    peers = rows if peers is None else peers
    out = []
    for idx, r in rows.iterrows():
        u = int(r["user"])
        base = gcontrib.loc[peers.index[peers["user"].to_numpy() == u]].mean()
        g = gcontrib.loc[idx]
        rel = (g - base).sort_values(ascending=False)
        pos = rel[(rel > 0) & (g[rel.index] > 0)].head(top)
        out.append({
            "item": int(r["item"]),
            "title": ctx["item_meta"].at[int(r["item"]), "title"],
            "reasons": [reason_for(k, u, int(r["item"]), ctx) for k in pos.index] or ["strong overall score"],
            "drivers": {GROUP_LABELS.get(k, k): round(float(v), 3)
                        for k, v in g.sort_values(ascending=False).head(4).items()},
        })
    return out
