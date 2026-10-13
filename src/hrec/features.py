"""Candidate generation (stage 1 union) and ranker features (stage 2 input)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import scipy.sparse as sp

from .data import Stage
from .metrics import mask_seen

# Which feature group each column belongs to; used for ablations and SHAP rollups.
SIDE_GROUPS = {"agreement", "item", "user", "taste"}


@dataclass
class ScoreBank:
    """Masked dense score + rank matrices for every retriever on one stage."""
    users: np.ndarray                 # users with a target at this stage
    scores: dict[str, np.ndarray]     # name -> (len(users), n_items), seen = -inf
    ranks: dict[str, np.ndarray]      # name -> 0-based rank of every item per user

    @classmethod
    def build(cls, stage: Stage, models: dict) -> "ScoreBank":
        users = np.where(stage.targets >= 0)[0]
        scores, ranks = {}, {}
        for name, m in models.items():
            s = mask_seen(m.score(users), stage.matrix(), users).astype(np.float32)
            order = np.argsort(-s, axis=1, kind="stable")
            r = np.empty_like(order, dtype=np.int32)
            np.put_along_axis(r, order, np.arange(s.shape[1], dtype=np.int32)[None, :], axis=1)
            scores[name], ranks[name] = s, r
        return cls(users, scores, ranks)


def candidate_mask(bank: ScoreBank, n_per: dict[str, int]) -> np.ndarray:
    """Union of each retriever's top-N (after masking seen items)."""
    C = None
    for name, n in n_per.items():
        hit = bank.ranks[name] < n
        C = hit if C is None else (C | hit)
    return C


def _user_genre_profiles(stage: Stage, decay=0.9):
    meta = stage.item_meta
    vocab = sorted({g for gs in meta["genres"] for g in gs}) or ["_none"]
    gi = {g: j for j, g in enumerate(vocab)}
    r, c = [], []
    for i, gs in enumerate(meta["genres"]):
        for g in gs:
            r.append(i), c.append(gi[g])
    G = normalize(sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(stage.n_items, len(vocab))))
    rows, cols, vals = [], [], []
    for u, seq in enumerate(stage.sequences()):
        if len(seq):
            rows.extend([u] * len(seq)), cols.extend(seq), vals.extend(decay ** np.arange(len(seq))[::-1])
    W = sp.csr_matrix((vals, (rows, cols)), shape=(stage.n_users, stage.n_items))
    return normalize(W @ G), G, vocab


def build_features(stage: Stage, bank: ScoreBank, C: np.ndarray, recent_pop: np.ndarray) -> pd.DataFrame:
    """One row per (user, candidate item). Rows are grouped by user (needed by LambdaRank)."""
    ui, items = np.nonzero(C)
    users = bank.users[ui]
    f = {"user": users, "item": items}

    # --- retriever signals: raw score, log-rank, and per-user z-score
    for name, s in bank.scores.items():
        finite = np.where(np.isfinite(s), s, np.nan)
        mu, sd = np.nanmean(finite, axis=1), np.nanstd(finite, axis=1) + 1e-9
        val = s[ui, items]
        f[f"{name}__score"] = val
        f[f"{name}__z"] = (val - mu[ui]) / sd[ui]
        f[f"{name}__logrank"] = np.log1p(bank.ranks[name][ui, items])
    f["agreement__n_top50"] = sum((bank.ranks[n][ui, items] < 50).astype(np.int8) for n in bank.ranks)

    # --- item side
    h = stage.history
    pop = stage.item_popularity()
    gmean = h["rating"].mean()
    agg = h.groupby("item")["rating"].agg(["sum", "count"]).reindex(range(stage.n_items), fill_value=0)
    bayes = ((agg["sum"] + 10 * gmean) / (agg["count"] + 10)).to_numpy()
    year = stage.item_meta["year"].to_numpy(dtype=float)
    f["item__logpop"] = np.log1p(pop[items])
    f["item__recentpop"] = np.log1p(recent_pop[items])
    f["item__bayes_rating"] = bayes[items]
    f["item__year"] = year[items]

    # --- user side
    hl = stage.history_lengths()
    ustat = h.groupby("user").agg(mean_rating=("rating", "mean"))
    ustat["mainstream"] = pd.Series(np.log1p(pop[h["item"].to_numpy()]), index=h.index).groupby(h["user"]).mean()
    ustat["year"] = pd.Series(year[h["item"].to_numpy()], index=h.index).groupby(h["user"]).mean()
    ustat = ustat.reindex(range(stage.n_users))
    um = stage.user_meta
    f["user__loghist"] = np.log1p(hl[users])
    f["user__mean_rating"] = ustat["mean_rating"].to_numpy()[users]
    f["user__mainstream"] = ustat["mainstream"].to_numpy()[users]
    f["user__age"] = um["age"].to_numpy(dtype=float)[users]
    f["user__occupation"] = pd.Categorical(um["occupation"].to_numpy()[users],
                                           categories=sorted(um["occupation"].unique()))

    # --- user x item "taste" crosses
    P, G, _ = _user_genre_profiles(stage)
    f["taste__genre_affinity"] = np.asarray(P[users].multiply(G[items]).sum(axis=1)).ravel()
    f["taste__pop_gap"] = f["item__logpop"] - f["user__mainstream"]
    f["taste__year_gap"] = f["item__year"] - ustat["year"].to_numpy()[users]

    df = pd.DataFrame(f)
    df["label"] = (stage.targets[users] == items).astype(np.int8)
    return df


def feature_columns(df: pd.DataFrame, drop_groups: set[str] = frozenset()) -> list[str]:
    cols = [c for c in df.columns if "__" in c]
    return [c for c in cols if c.split("__")[0] not in drop_groups]
