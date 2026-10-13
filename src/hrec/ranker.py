"""Stage-2 ranking: a gradient-boosted ranker over retriever signals + side features,
plus a tuned linear blend as the "classic hybrid" baseline it has to beat.

Why gradient-boosted trees and not a deep CTR model? With a few thousand training
queries, a deep ranker would mostly learn noise. GBDT rankers are still what many
production search/recsys stacks use at this stage, they take mixed-scale features
without normalisation, and they come with exact, fast SHAP values (TreeSHAP), which
is what makes the explanations here faithful rather than decorative.

Ranker settings are chosen on the most recent training slice (see _temporal_holdout),
never by looking at test.
"""
from __future__ import annotations

import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import feature_columns

GRID = {
    "num_leaves": [7, 31],
    "min_data_in_leaf": [100, 400],
    "raw_scores": [True, False],
    "init_blend": [False, True],
}
RETRIEVERS = ["pop", "itemknn", "ease", "content", "lightgcn", "sasrec"]


def _groups(q: np.ndarray) -> np.ndarray:
    """Rows are contiguous per query; return run lengths."""
    change = np.flatnonzero(np.diff(q)) + 1
    return np.diff(np.concatenate([[0], change, [len(q)]]))


def _with_positives(df: pd.DataFrame) -> pd.DataFrame:
    # A query with no positive among its candidates carries no ranking signal
    df = df if "qid" in df else df.assign(qid=df["user"])
    return df[df.groupby("qid")["label"].transform("max") > 0]


def _temporal_holdout(d: pd.DataFrame, seed: int) -> np.ndarray:
    """Validate on the most recent training slice (slice 0 = the tune stage), train on
    the older ones. Each slice has its own independently trained retrievers, so this
    reproduces the shift the ranker faces at test time (new retriever instances,
    different score scales). A random user holdout inside the same slices would hide it.
    """
    if "slice" in d and d["slice"].nunique() > 1:
        return (d["slice"] == d["slice"].min()).to_numpy()
    users = np.sort(d["user"].unique())
    va_users = np.random.default_rng(seed).choice(users, size=max(1, len(users) // 5), replace=False)
    return d["user"].isin(va_users).to_numpy()


class LambdaRanker:
    """LightGBM LambdaRank.

    raw_scores=False drops each retriever's raw score and keeps only scale-free versions
    (per-user z-score, log-rank). init_blend=True starts boosting from the tuned linear
    blend (LightGBM init_score), so the trees only learn corrections to it.
    """

    def __init__(self, drop_groups=frozenset(), seed: int = 0, verbose: bool = False,
                 num_leaves=31, min_data_in_leaf=100, raw_scores=True, init_blend=False,
                 learning_rate=0.02, objective="lambdarank"):
        self.drop_groups, self.seed, self.verbose = set(drop_groups), seed, verbose
        self.raw_scores, self.init_blend = raw_scores, init_blend
        self.retrievers = [r for r in RETRIEVERS if r not in self.drop_groups]
        self.params = dict(objective=objective, metric="ndcg", eval_at=[10], learning_rate=learning_rate,
                           num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf,
                           bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
                           lambdarank_truncation_level=20, verbose=-1, seed=seed, num_threads=2)

    def _ds(self, d, blend, ref=None):
        init = blend.predict(d) if blend is not None else None
        return lgb.Dataset(d[self.cols], d["label"], group=_groups(d["qid"].to_numpy()),
                           init_score=init, reference=ref, free_raw_data=False)

    def fit(self, df: pd.DataFrame, blend_tr: "LinearBlend | None" = None) -> "LambdaRanker":
        cols = feature_columns(df, self.drop_groups)
        self.cols = cols if self.raw_scores else [c for c in cols if not c.endswith("__score")]
        d = _with_positives(df)
        va = _temporal_holdout(d, self.seed)
        b_tr = None
        if self.init_blend:
            b_tr = blend_tr or LinearBlend(self.retrievers, seed=self.seed).fit(d[~va])
        dtr = self._ds(d[~va], b_tr)
        dva = self._ds(d[va], b_tr, ref=dtr)
        b = lgb.train(self.params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        self.best_iter = max(int(b.best_iteration or 0), 1)
        self.val_ndcg = float(b.best_score["valid_0"]["ndcg@10"])
        # refit on every training query (blend too) with the early-stopped number of trees
        self.blend = LinearBlend(self.retrievers, seed=self.seed).fit(d) if self.init_blend else None
        self.booster = lgb.train(self.params, self._ds(d, self.blend), num_boost_round=self.best_iter)
        if self.verbose:
            print(f"    ranker: {len(self.cols)} features, {self.best_iter} trees, "
                  f"{d['qid'].nunique()} queries, val NDCG@10 {self.val_ndcg:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base = self.blend.predict(df) if self.blend is not None else 0.0
        return base + self.booster.predict(df[self.cols])

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact additive attribution of each candidate's score.

        Tree part: TreeSHAP (Lundberg et al. 2020) via LightGBM's native pred_contrib.
        Blend part (if init_blend): w_r * z_r per retriever, already additive.
        Row sums (incl. bias) reproduce predict() exactly.
        """
        c = self.booster.predict(df[self.cols], pred_contrib=True)
        out = pd.DataFrame(c, columns=self.cols + ["bias"], index=df.index)
        if self.blend is not None:
            for r, w in zip(self.blend.retrievers, self.blend.w):
                out[f"{r}__blend"] = w * df[f"{r}__z"].to_numpy()
        return out


def select_ranker(df: pd.DataFrame, seed: int = 0) -> tuple[dict, list[dict]]:
    """Grid search scored by NDCG@10 on the held-out (most recent) training slice."""
    d = _with_positives(df)
    va = _temporal_holdout(d, seed)
    blend_tr = LinearBlend(RETRIEVERS, seed=seed).fit(d[~va])
    rows = []
    for vals in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, vals))
        r = LambdaRanker(seed=seed, **kw).fit(df, blend_tr=blend_tr)
        rows.append(kw | {"val_ndcg10": round(r.val_ndcg, 4), "trees": r.best_iter})
    # reference point: the blend alone on the same validation slice
    rows.append({"num_leaves": "-", "min_data_in_leaf": "-", "raw_scores": "-", "init_blend": "blend only",
                 "val_ndcg10": round(blend_tr.score(d[va]), 4), "trees": 0})
    best = max((r for r in rows if r["trees"]), key=lambda r: r["val_ndcg10"])
    return {k: best[k] for k in GRID}, rows


class LinearBlend:
    """Weighted sum of per-user z-scored retriever scores — the v1-style hybrid, but with
    weights searched on training slices (random search on the simplex)."""

    def __init__(self, retrievers: list[str], n_trials: int = 400, seed: int = 0):
        self.retrievers, self.n_trials, self.seed = list(retrievers), n_trials, seed

    @staticmethod
    def _ndcg_fast(qid, label, s):
        """NDCG@10 with one positive per query: rank = #candidates scoring >= the positive."""
        starts = np.concatenate([[0], np.flatnonzero(np.diff(qid)) + 1])
        sizes = np.diff(np.concatenate([starts, [len(qid)]]))
        pos_s = np.full(len(starts), np.inf)
        qi = np.repeat(np.arange(len(starts)), sizes)
        has = label == 1
        pos_s[qi[has]] = s[has]
        r = np.add.reduceat((s >= pos_s[qi]).astype(np.int32), starts) - 1
        ok = np.isfinite(pos_s)
        return float(np.where(ok & (r < 10), 1 / np.log2(np.maximum(r, 0) + 2), 0).sum() / len(starts))

    def _Z(self, df):
        return df[[f"{r}__z" for r in self.retrievers]].to_numpy()

    def fit(self, df):
        q = df["qid"].to_numpy() if "qid" in df else df["user"].to_numpy()
        y, Z = df["label"].to_numpy(), self._Z(df)
        rng = np.random.default_rng(self.seed)
        W = rng.dirichlet(np.ones(len(self.retrievers)) * 0.7, size=self.n_trials)
        W = np.vstack([np.eye(len(self.retrievers)), W])  # include each single retriever
        scores = [self._ndcg_fast(q, y, Z @ w) for w in W]
        i = int(np.argmax(scores))
        self.val_ndcg, self.w = scores[i], W[i]
        return self

    def score(self, df):
        q = df["qid"].to_numpy() if "qid" in df else df["user"].to_numpy()
        return self._ndcg_fast(q, df["label"].to_numpy(), self.predict(df))

    def predict(self, df):
        return self._Z(df) @ self.w

    def weights(self):
        return {r: round(float(w), 3) for r, w in zip(self.retrievers, self.w)}
