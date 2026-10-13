"""Approximate nearest-neighbour retrieval with HNSW.

LightGCN and SASRec both score with a dot product between a user vector and item
vectors, so "top-N for this user" is a maximum-inner-product search. Brute force is
O(items) per request; HNSW (Malkov & Yashunin, 2018) is a navigable small-world graph
that gets there in roughly O(log items) hops at the cost of a small recall loss.
At 1.7k items brute force is fine (the experiment uses it); this module is what you'd
swap in at catalog sizes where it isn't, and scripts/bench_ann.py measures the trade.
"""
from __future__ import annotations

import hnswlib
import numpy as np


class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, threads: int = 1):
        self.dim, self.M, self.efc, self.threads = dim, M, ef_construction, threads

    def build(self, item_vecs: np.ndarray) -> "HNSWIndex":
        self.index = hnswlib.Index(space="ip", dim=self.dim)
        self.index.init_index(max_elements=len(item_vecs), M=self.M, ef_construction=self.efc)
        self.index.set_num_threads(self.threads)
        self.index.add_items(item_vecs.astype(np.float32), np.arange(len(item_vecs)))
        return self

    def query(self, user_vecs: np.ndarray, k: int, ef: int = 200) -> np.ndarray:
        self.index.set_ef(max(ef, k))
        labels, _ = self.index.knn_query(user_vecs.astype(np.float32), k=k)
        return labels


def exact_topk(user_vecs: np.ndarray, item_vecs: np.ndarray, k: int, batch: int = 256) -> np.ndarray:
    out = []
    for s in range(0, len(user_vecs), batch):
        sc = user_vecs[s: s + batch] @ item_vecs.T
        idx = np.argpartition(-sc, k, axis=1)[:, :k]
        out.append(idx)
    return np.vstack(out)


def recall_at_k(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]))
