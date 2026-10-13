"""Non-neural retrievers: popularity, item-kNN, EASE, and a content model.

EASE deserves a note. It's a linear item-item autoencoder with a closed-form solution
(Steck, WWW 2019): one matrix inverse, no SGD, no epochs. On MovieLens-scale data it
routinely matches or beats deep models, which makes it the baseline any fancier model
has to clear before it's allowed into the system.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data import Stage
from .base import Retriever


class Popularity(Retriever):
    """Recency-weighted popularity: recent interactions count more (half-life in days)."""

    name = "pop"

    def __init__(self, half_life_days: float = 30.0):
        self.half_life_days = half_life_days

    def fit(self, stage: Stage):
        h = stage.history
        age_days = (h["ts"].max() - h["ts"].to_numpy()) / 86400.0
        w = 0.5 ** (age_days / self.half_life_days)
        self.pop = np.bincount(h["item"], weights=w, minlength=stage.n_items)
        return self

    def score(self, users):
        return np.tile(np.log1p(self.pop), (len(users), 1))

    def params(self):
        return {"half_life_days": self.half_life_days}


class ItemKNN(Retriever):
    """Classic item-based CF: cosine similarity on co-occurrence, top-k neighbours kept."""

    name = "itemknn"

    def __init__(self, k: int = 100, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, stage: Stage):
        X = stage.matrix().astype(np.float64)
        self.X = X.tocsr()
        co = (X.T @ X).toarray()
        n = np.sqrt(np.diag(co))
        S = co / (np.outer(n, n) + self.shrink + 1e-12)
        np.fill_diagonal(S, 0.0)
        if self.k < S.shape[0]:
            thresh = -np.partition(-S, self.k - 1, axis=0)[self.k - 1]
            S[S < thresh[None, :]] = 0.0
        self.S = S
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.S)

    def params(self):
        return {"k": self.k, "shrink": self.shrink}


class EASE(Retriever):
    """Embarrassingly Shallow Autoencoder. B = argmin ||X - XB||² + λ||B||², diag(B)=0."""

    name = "ease"

    def __init__(self, lam: float = 250.0):
        self.lam = lam

    def fit(self, stage: Stage):
        X = stage.matrix().astype(np.float64)
        self.X = X.tocsr()
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B = B
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.B)

    def params(self):
        return {"lambda": self.lam}


class Content(Retriever):
    """Content-based: item vectors from genres + title TF-IDF + release decade.

    A user's profile is a recency-weighted average of the items they touched, so the
    last few items matter most. This is the one retriever that still works when a user
    has 1-2 interactions or an item has zero, which is exactly why it's here.
    """

    name = "content"

    def __init__(self, decay: float = 0.9, w_genre=1.0, w_title=0.5, w_decade=0.5):
        self.decay, self.w = decay, (w_genre, w_title, w_decade)

    def item_features(self, stage: Stage) -> sp.csr_matrix:
        meta = stage.item_meta
        vocab = sorted({g for gs in meta["genres"] for g in gs})
        gi = {g: j for j, g in enumerate(vocab)}
        rows, cols = [], []
        for i, gs in enumerate(meta["genres"]):
            for g in gs:
                rows.append(i), cols.append(gi[g])
        G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(meta), max(len(vocab), 1)))
        titles = meta["title"].fillna("").tolist()
        try:
            T = TfidfVectorizer(min_df=2, stop_words="english").fit_transform(titles)
        except ValueError:  # empty vocabulary (e.g. tiny test data)
            T = sp.csr_matrix((len(meta), 1))
        dec = (meta["year"].fillna(-1) // 10).astype(int).to_numpy()
        _, dec_codes = np.unique(dec, return_inverse=True)
        D = sp.csr_matrix((np.ones(len(meta)), (np.arange(len(meta)), dec_codes)))
        wg, wt, wd = self.w
        F = sp.hstack([wg * normalize(G), wt * normalize(T), wd * D]).tocsr()
        return normalize(F)

    def fit(self, stage: Stage):
        self.F = self.item_features(stage)
        rows, cols, vals = [], [], []
        for u, seq in enumerate(stage.sequences()):
            if len(seq):
                w = self.decay ** np.arange(len(seq))[::-1]
                rows.extend([u] * len(seq)), cols.extend(seq), vals.extend(w)
        W = sp.csr_matrix((vals, (rows, cols)), shape=(stage.n_users, stage.n_items))
        self.P = normalize(W @ self.F)
        return self

    def score(self, users):
        return np.asarray((self.P[users] @ self.F.T).todense())

    def params(self):
        return {"decay": self.decay, "weights(genre,title,decade)": self.w}
