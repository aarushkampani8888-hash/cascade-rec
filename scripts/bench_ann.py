"""Exact vs HNSW retrieval: recall@100 and per-query latency as the catalog grows.

Item vectors are synthetic, because the repo's catalog (1.7k items) is far too small for
ANN to matter. They're built to look like learned embeddings: a clustered cloud on a
low-dimensional subspace (intrinsic dim 16) embedded in 64-d, plus a little noise.
That matters: on isotropic high-dimensional noise, every graph index collapses, and on
real embeddings it doesn't. Single-threaded queries: latency = one request, one core.

    python scripts/bench_ann.py --sizes 10000 100000 300000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrec.ann import HNSWIndex, exact_topk, recall_at_k  # noqa: E402


BASIS = {}


def clustered(n, d, rng, k=256, intrinsic=16):
    if d not in BASIS:  # shared subspace + cluster centres for items and users
        BASIS[d] = (np.linalg.qr(rng.normal(size=(d, intrinsic)))[0].T, rng.normal(size=(k, intrinsic)))
    P, centers = BASIS[d]
    z = centers[rng.integers(0, k, n)] + 0.5 * rng.normal(size=(n, intrinsic))
    x = z @ P + 0.05 * rng.normal(size=(n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


p = argparse.ArgumentParser()
p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 300_000])
p.add_argument("--dim", type=int, default=64)
p.add_argument("--queries", type=int, default=500)
p.add_argument("--k", type=int, default=100)
p.add_argument("--out", default="results/ann_bench.json")
a = p.parse_args()

rng = np.random.default_rng(0)
rows = []
for n in a.sizes:
    items = clustered(n, a.dim, rng)
    users = clustered(a.queries, a.dim, rng) * rng.uniform(0.5, 2.0, size=(a.queries, 1)).astype(np.float32)
    t = time.perf_counter()
    exact = np.vstack([exact_topk(users[i:i + 1], items, a.k) for i in range(a.queries)])
    exact_ms = (time.perf_counter() - t) / a.queries * 1e3
    t = time.perf_counter()
    idx = HNSWIndex(a.dim, threads=2).build(items)
    build_s = time.perf_counter() - t
    idx.threads = 1
    idx.index.set_num_threads(1)
    for ef in (100, 200, 400):
        t = time.perf_counter()
        approx = np.vstack([idx.query(users[i:i + 1], a.k, ef) for i in range(a.queries)])
        ms = (time.perf_counter() - t) / a.queries * 1e3
        r = {"items": n, "ef": ef, "recall@100": round(recall_at_k(approx, exact), 4),
             "hnsw_ms": round(ms, 3), "exact_ms": round(exact_ms, 3),
             "speedup": round(exact_ms / ms, 1), "build_s": round(build_s, 1)}
        rows.append(r)
        print(r, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(rows, indent=2))
