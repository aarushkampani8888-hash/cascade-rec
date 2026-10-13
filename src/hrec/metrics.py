"""Full-catalog ranking metrics.

We rank the target against *every* item the user hasn't seen, not against 100 sampled
negatives. Sampled metrics can reorder models (Krichene & Rendle, KDD 2020), and the
catalog here is small enough that there's no excuse.

Ties are broken pessimistically: an item tied with the target counts as ranked above
it. That stops a model that outputs constant scores from looking good by luck.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def mask_seen(scores: np.ndarray, seen: sp.csr_matrix, users: np.ndarray) -> np.ndarray:
    """Set already-seen items to -inf (in place on a copy)."""
    s = np.array(scores, dtype=np.float64, copy=True)
    sub = seen[users]
    rows = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
    s[rows, sub.indices] = -np.inf
    return s


def target_ranks(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """0-based rank of each user's target among their scored items (pessimistic ties)."""
    t = scores[np.arange(len(targets)), targets][:, None]
    if np.any(~np.isfinite(t)):
        # Target outside candidate set: it can't be recommended -> worst rank.
        r = (scores >= t).sum(axis=1) - 1
        r[~np.isfinite(t[:, 0])] = scores.shape[1]
        return r
    return (scores >= t).sum(axis=1) - 1


def per_user_metrics(ranks: np.ndarray, ks=(10, 20)) -> dict[str, np.ndarray]:
    out = {}
    for k in ks:
        hit = ranks < k
        out[f"Recall@{k}"] = hit.astype(float)            # one target -> recall == hit rate
        out[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(ranks + 2), 0.0)
    out["MRR"] = 1.0 / (ranks + 1)
    return out


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def beyond_accuracy(topk: np.ndarray, item_pop: np.ndarray, n_users_hist: int) -> dict:
    """Catalog coverage and novelty (mean self-information of recommended items)."""
    n_items = len(item_pop)
    coverage = len(np.unique(topk)) / n_items
    p = (item_pop + 1) / (n_users_hist + 1)
    novelty = float(np.mean(-np.log2(p[topk])))
    return {"Coverage@10": coverage, "Novelty@10": novelty}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict:
    """95% CI for mean(a - b) by resampling users. Paired, so per-user difficulty cancels."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((means <= 0).mean())}


def bootstrap_ci(a: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)
