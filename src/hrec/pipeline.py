"""End-to-end experiment: retrieve -> rank -> evaluate -> explain.

Two phases so the cheap part can be iterated without retraining neural nets:

* `build()`  — split, fit/tune retrievers, score, generate candidates, build features.
               Expensive (~30 min on 2 CPU cores for ML-100K). Cached to results/cache.pkl.
* `rank_and_report()` — select ranker settings on *validation*, fit, evaluate on test,
               ablate, explain. A minute or two.

Read top to bottom; each numbered block is one step of the protocol.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .data import Split, Stage, leave_last_out, load
from .explain import explain, group_contributions
from .features import ScoreBank, build_features, candidate_mask
from .ranker import LambdaRanker, LinearBlend, select_ranker
from .retrievers import EASE, Content, ItemKNN, LightGCN, Popularity, SASRec

RETRIEVERS = ["pop", "itemknn", "ease", "content", "lightgcn", "sasrec"]
SIDE = {"agreement", "item", "user", "taste"}


@dataclass
class Config:
    dataset: str = "ml-100k"
    data_root: str = "data"
    out: str = "results"
    seed: int = 0
    cold_frac: float = 0.2
    cold_keep: int = 5
    ease_lambdas: tuple = (50, 100, 250, 500, 1000)
    n_candidates: dict = field(default_factory=lambda: {
        "ease": 150, "sasrec": 150, "lightgcn": 150, "itemknn": 50, "content": 50, "pop": 25})
    sasrec_max_epochs: int = 300
    lightgcn_max_epochs: int = 400
    ranker_slices: int = 3  # time slices used to train the ranker (offsets 1..k)
    ablations: bool = True
    threads: int = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- helpers
def evaluate(scores: np.ndarray, stage: Stage, users: np.ndarray) -> dict:
    ranks = M.target_ranks(scores, stage.targets[users])
    per_user = M.per_user_metrics(ranks)
    topk = M.top_k(scores, 10)
    extra = M.beyond_accuracy(topk, stage.item_popularity(), stage.n_users)
    return {"per_user": per_user, **extra}


def scores_from_rows(df: pd.DataFrame, pred: np.ndarray, users: np.ndarray, n_items: int) -> np.ndarray:
    S = np.full((len(users), n_items), -np.inf)
    S[np.searchsorted(users, df["user"].to_numpy()), df["item"].to_numpy()] = pred
    return S


def make_retrievers(cfg: Config, tuned: dict, stage_name: str) -> dict:
    v = stage_name == "tune"  # only the tune stage early-stops; every other stage uses fixed epochs
    return {
        "pop": Popularity(),
        "itemknn": ItemKNN(),
        "ease": EASE(tuned.get("ease_lambda", 250)),
        "content": Content(),
        "lightgcn": LightGCN(seed=cfg.seed, max_epochs=cfg.lightgcn_max_epochs,
                             epochs=None if v else tuned["lightgcn_epochs"]),
        "sasrec": SASRec(seed=cfg.seed, max_epochs=cfg.sasrec_max_epochs,
                         epochs=None if v else tuned["sasrec_epochs"]),
    }


def in_candidates(F: pd.DataFrame, n_per: dict) -> np.ndarray:
    """Recover 'was this row retrieved by any of these retrievers' from logrank features."""
    keep = np.zeros(len(F), dtype=bool)
    for n, k in n_per.items():
        keep |= F[f"{n}__logrank"].to_numpy() < np.log1p(k)
    return keep


# ----------------------------------------------------------------------------- phase 1
def build(cfg: Config, ds=None) -> dict:
    torch.set_num_threads(cfg.threads)
    t0 = time.time()
    R = {"config": asdict(cfg)}

    # 1. data + split -----------------------------------------------------------
    ds = ds if ds is not None else load(cfg.dataset, cfg.data_root)
    split: Split = leave_last_out(ds, cfg.cold_frac, cfg.cold_keep, cfg.seed)
    A, B = split.tune, split.test
    R["data"] = ds.describe() | {"cold_users": int(split.cold_users.sum())}
    log(f"data {R['data']}")

    # 2. stage A: fit retrievers on history, tune on val ----------------------------
    tuned, val_scores = {}, {}
    users_A = np.where(A.targets >= 0)[0]
    best = (-1, None)
    for lam in cfg.ease_lambdas:
        m = EASE(lam).fit(A)
        r = M.target_ranks(M.mask_seen(m.score(users_A), A.matrix(), users_A), A.targets[users_A])
        best = max(best, (M.per_user_metrics(r)["NDCG@10"].mean(), lam))
    tuned["ease_lambda"] = best[1]
    log(f"EASE lambda={best[1]} (val NDCG@10 {best[0]:.4f})")

    models_A = make_retrievers(cfg, tuned, "tune")
    for name, m in models_A.items():
        t = time.time()
        if name in ("lightgcn", "sasrec"):
            m.fit(A, A.targets)  # early stopping on VALIDATION targets
            tuned[f"{name}_epochs"] = m.best_epoch
        else:
            m.fit(A)
        log(f"fit {name} [tune] in {time.time() - t:.0f}s")
    bank_A = ScoreBank.build(A, models_A)
    for name, s in bank_A.scores.items():
        val_scores[name] = float(evaluate(s, A, bank_A.users)["per_user"]["NDCG@10"].mean())
    R["tuned"], R["val_ndcg10"] = tuned, val_scores

    # 3. stage B: refit everything on history+val with the tuned settings --------------
    models_B = make_retrievers(cfg, tuned, "test")
    for name, m in models_B.items():
        t = time.time()
        m.fit(B)  # no targets passed: nothing about the test items is visible
        log(f"fit {name} [test] in {time.time() - t:.0f}s")
    bank_B = ScoreBank.build(B, models_B)
    users_B = bank_B.users
    cold_B = split.cold_users[users_B]
    R["params"] = {n: m.params() for n, m in models_B.items()}
    single = {name: evaluate(s, B, users_B) for name, s in bank_B.scores.items()}

    # 4. candidates -------------------------------------------------------------------
    C_B = candidate_mask(bank_B, cfg.n_candidates)
    idx = np.arange(len(users_B)), B.targets[users_B]
    R["candidates"] = {
        "mean_per_user": float(C_B.sum(1).mean()),
        "recall_union": float(C_B[idx].mean()),
        "recall_union_cold": float(C_B[idx][cold_B].mean()),
        "recall_per_retriever": {n: float((bank_B.ranks[n][idx] < k).mean()) for n, k in cfg.n_candidates.items()},
    }
    log(f"candidates: {R['candidates']['mean_per_user']:.0f}/user, union recall {R['candidates']['recall_union']:.3f}")

    # 5. ranker training data: tune slice + earlier slices ------------------------------
    # Each slice refits the retrievers on that slice's history (fixed tuned epochs), so
    # the ranker sees retriever scores produced exactly the way they'll be at test time.
    slices = [(A, models_A, bank_A)]
    for off in range(2, cfg.ranker_slices + 1):
        St = split.stage_at(off)
        ms = make_retrievers(cfg, tuned, St.name)
        for m in ms.values():
            m.fit(St)
        slices.append((St, ms, ScoreBank.build(St, ms)))
        log(f"built ranker-training slice {St.name}")
    parts = []
    for k, (St, ms, bk) in enumerate(slices):
        f = build_features(St, bk, candidate_mask(bk, cfg.n_candidates), ms["pop"].pop)
        f.insert(0, "qid", k * St.n_users + f["user"].to_numpy())
        f.insert(1, "slice", k)
        parts.append(f)
    F_A = pd.concat(parts, ignore_index=True)
    F_B = build_features(B, bank_B, C_B, models_B["pop"].pop)
    F_B.insert(0, "qid", F_B["user"].to_numpy())
    log(f"features: train {F_A.shape} ({F_A['qid'].nunique()} queries), test {F_B.shape}")

    R["build_sec"] = round(time.time() - t0, 1)
    return {"R": R, "single": single, "F_A": F_A, "F_B": F_B, "B": B, "users_B": users_B,
            "cold_users": split.cold_users, "raw_user_ids": ds.raw_user_ids,
            "ctx": {"item_meta": B.item_meta, "sequences": B.sequences(),
                    "itemknn_S": models_B["itemknn"].S.astype(np.float16)}}  # only used to pick anchors


# ----------------------------------------------------------------------------- phase 2
def rank_and_report(cfg: Config, cache: dict) -> dict:
    t0 = time.time()
    R = dict(cache["R"])
    F_A, F_B, B, users_B = cache["F_A"], cache["F_B"].copy(), cache["B"], cache["users_B"]
    if "slice" not in F_A:
        F_A = F_A.assign(slice=F_A["qid"] // B.n_users)
    cold_users = cache["cold_users"]
    cold_B = cold_users[users_B]
    evals = dict(cache["single"])

    # 6. pick ranker settings on validation queries, then fit --------------------------
    best_kw, grid = select_ranker(F_A, seed=cfg.seed)
    R["ranker_grid"] = grid
    R["ranker_settings"] = best_kw
    log(f"ranker settings (chosen on validation): {best_kw}")
    ranker = LambdaRanker(seed=cfg.seed, verbose=True, **best_kw).fit(F_A)
    F_B["score"] = ranker.predict(F_B)
    evals["hybrid (learned ranker)"] = evaluate(scores_from_rows(F_B, F_B["score"].to_numpy(), users_B, B.n_items), B, users_B)
    R["ranker_trees"] = ranker.best_iter

    blend = LinearBlend(RETRIEVERS, seed=cfg.seed).fit(F_A)
    evals["linear blend (tuned)"] = evaluate(scores_from_rows(F_B, blend.predict(F_B), users_B, B.n_items), B, users_B)
    R["blend_weights"] = blend.weights()

    # 7. ablations: drop one source at a time (its candidates AND its features) -------
    R["ablations"] = {}
    if cfg.ablations:
        full = evals["hybrid (learned ranker)"]["per_user"]["NDCG@10"]
        for drop in RETRIEVERS + ["side"]:
            fa, fb = F_A, F_B
            if drop == "side":
                groups = SIDE
            else:
                groups = {drop}
                rest = {k: v for k, v in cfg.n_candidates.items() if k != drop}
                fa, fb = F_A[in_candidates(F_A, rest)], F_B[in_candidates(F_B, rest)]
                names = [n for n in RETRIEVERS if n != drop]  # agreement must not peek at the dropped one
                fa, fb = (f.assign(agreement__n_top50=sum((f[f"{n}__logrank"] < np.log(51)).astype(np.int8)
                                                          for n in names)) for f in (fa, fb))
            rk = LambdaRanker(drop_groups=groups, seed=cfg.seed, **best_kw).fit(fa)
            nd = evaluate(scores_from_rows(fb, rk.predict(fb), users_B, B.n_items), B, users_B)["per_user"]["NDCG@10"]
            R["ablations"][drop] = {"NDCG@10": float(nd.mean()),
                                    "delta_vs_full": M.paired_bootstrap(nd, full, seed=cfg.seed),
                                    "NDCG@10_cold": float(nd[cold_B].mean())}
            log(f"ablation -{drop}: NDCG@10 {nd.mean():.4f}")

    # 8. results table + significance + cold/warm slices --------------------------------
    hyb = evals["hybrid (learned ranker)"]["per_user"]
    table = []
    for name, e in evals.items():
        pu = e["per_user"]
        row = {"model": name}
        for k in ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MRR"]:
            row[k] = float(pu[k].mean())
        row["NDCG@10_ci"] = M.bootstrap_ci(pu["NDCG@10"], seed=cfg.seed)
        row["NDCG@10_warm"] = float(pu["NDCG@10"][~cold_B].mean())
        row["NDCG@10_cold"] = float(pu["NDCG@10"][cold_B].mean())
        row["Coverage@10"], row["Novelty@10"] = e["Coverage@10"], e["Novelty@10"]
        if name != "hybrid (learned ranker)":
            row["hybrid_minus_this"] = M.paired_bootstrap(hyb["NDCG@10"], pu["NDCG@10"], seed=cfg.seed)
            row["hybrid_minus_this_cold"] = M.paired_bootstrap(hyb["NDCG@10"][cold_B], pu["NDCG@10"][cold_B], seed=cfg.seed)
        table.append(row)
    R["results"] = table

    # 9. explanations: TreeSHAP on each user's top-20 -----------------------------------
    F_B["rank"] = F_B.groupby("user")["score"].rank(ascending=False, method="first")
    top = F_B[F_B["rank"] <= 20].copy()
    contrib = ranker.contributions(top)
    gc = group_contributions(contrib)
    trees_only = group_contributions(contrib[[c for c in contrib.columns if not c.endswith("__blend")]])
    is_cold = cold_users[top["user"].to_numpy()]
    imp = lambda g, m: g[m].abs().mean().sort_values(ascending=False).round(4).to_dict()
    R["shap_group_importance"] = {"warm": imp(gc, ~is_cold), "cold": imp(gc, is_cold)}
    R["tree_correction_importance"] = {"warm": imp(trees_only, ~is_cold), "cold": imp(trees_only, is_cold)}
    ctx = cache["ctx"]
    examples = {}
    for u in [int(users_B[~cold_B][0]), int(users_B[cold_B][0])]:
        rows = top[(top["user"] == u) & (top["rank"] <= 5)].sort_values("rank")
        examples[str(cache["raw_user_ids"][u])] = {
            "cold": bool(cold_users[u]),
            "recent_history": [ctx["item_meta"].at[i, "title"] for i in ctx["sequences"][u][-5:]],
            "recommendations": explain(rows, gc, ctx, peers=top[top["user"] == u])}
    R["examples"] = examples

    # 10. save -------------------------------------------------------------------------
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    R["rank_sec"] = round(time.time() - t0, 1)
    R["runtime_sec"] = R.get("build_sec", 0) + R["rank_sec"]
    (out / "metrics.json").write_text(json.dumps(R, indent=2, default=_json))
    np.savez_compressed(out / "per_user.npz", users=users_B, cold=cold_B,
                        **{f"ndcg10::{k}": e["per_user"]["NDCG@10"] for k, e in evals.items()})
    pd.to_pickle({"top": top, "group_contrib": gc, "ctx": ctx, "cold_users": cold_users,
                  "raw_user_ids": cache["raw_user_ids"]}, out / "explanations.pkl.gz")
    log(f"done -> {out}/")
    return R


def run(cfg: Config, ds=None, cache_path: str | Path | None = None) -> dict:
    cache = None
    if cache_path and Path(cache_path).exists():
        cache = pd.read_pickle(cache_path)
        log(f"loaded cached retrieval/features from {cache_path}")
    if cache is None:
        cache = build(cfg, ds)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(cache, cache_path)
    return rank_and_report(cfg, cache)


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
