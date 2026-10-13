"""Data loading and leakage-safe splitting.

Everything downstream works on contiguous integer ids (users 0..U-1, items 0..I-1)
and a single interaction table with columns: user, item, rating, ts.

The split protocol is *leave-last-out by time* per user:

    [ ... history ... | val | test ]

* Stage A ("tune"): models see `history`, are scored on `val`.
  Hyperparameters, early stopping and the learned ranker are fit here.
* Stage B ("test"): models are refit on `history + val`, scored on `test`.
  The test item is never touched until the final evaluation.

This is the main thing the v1 code got wrong: it picked the best alpha/beta/gamma
blend *on the test set*, which inflates every number it reports.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ML_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


@dataclass
class Dataset:
    name: str
    interactions: pd.DataFrame  # user, item, rating, ts  (contiguous int ids)
    item_meta: pd.DataFrame     # index=item; title:str, year:float, genres:list[str]
    user_meta: pd.DataFrame     # index=user; age:float, occupation:str
    raw_user_ids: np.ndarray
    raw_item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.raw_user_ids)

    @property
    def n_items(self) -> int:
        return len(self.raw_item_ids)

    def describe(self) -> dict:
        n = len(self.interactions)
        return {
            "dataset": self.name,
            "users": self.n_users,
            "items": self.n_items,
            "interactions": n,
            "density": round(n / (self.n_users * self.n_items), 5),
        }


# --------------------------------------------------------------------------- loaders
def _encode(name, df, user_meta_raw, item_meta_raw) -> Dataset:
    """Map raw ids to contiguous ints, dedupe (keep first touch), align metadata."""
    df = df.dropna(subset=["user", "item"]).copy()
    df = df.sort_values(["ts"], kind="stable")
    df = df.drop_duplicates(["user", "item"], keep="first")
    u_codes, u_raw = pd.factorize(df["user"])
    i_codes, i_raw = pd.factorize(df["item"])
    out = pd.DataFrame({
        "user": u_codes.astype(np.int64),
        "item": i_codes.astype(np.int64),
        "rating": df["rating"].astype(float).to_numpy(),
        "ts": df["ts"].astype(np.int64).to_numpy(),
    })

    im = pd.DataFrame(index=pd.Index(i_raw, name="raw"))
    if item_meta_raw is not None and len(item_meta_raw):
        im = im.join(item_meta_raw, how="left")
    im["title"] = im.get("title", pd.Series(index=im.index, dtype=object)).fillna("").astype(str)
    im["year"] = pd.to_numeric(im.get("year", pd.Series(index=im.index, dtype=float)), errors="coerce")
    g = im.get("genres", pd.Series(index=im.index, dtype=object))
    im["genres"] = [x if isinstance(x, list) else [] for x in g]
    im = im.reset_index(drop=True)
    im.index.name = "item"

    um = pd.DataFrame(index=pd.Index(u_raw, name="raw"))
    if user_meta_raw is not None and len(user_meta_raw):
        um = um.join(user_meta_raw, how="left")
    um["age"] = pd.to_numeric(um.get("age", pd.Series(index=um.index, dtype=float)), errors="coerce")
    um["occupation"] = um.get("occupation", pd.Series(index=um.index, dtype=object)).fillna("unknown").astype(str)
    um = um.reset_index(drop=True)
    um.index.name = "user"
    return Dataset(name, out, im[["title", "year", "genres"]], um[["age", "occupation"]],
                   np.asarray(u_raw), np.asarray(i_raw))


def load_movielens_100k(path: str | Path) -> Dataset:
    path = Path(path)
    df = pd.read_csv(path / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
    cols = ["item", "title", "release", "video", "url"] + ML_GENRES
    items = pd.read_csv(path / "u.item", sep="|", names=cols, encoding="latin-1")
    items["year"] = items["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    items["title"] = items["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    items["genres"] = [[g for g in ML_GENRES if row[g] == 1 and g != "unknown"]
                       for _, row in items.iterrows()]
    items = items.set_index("item")[["title", "year", "genres"]]
    users = pd.read_csv(path / "u.user", sep="|", names=["user", "age", "gender", "occupation", "zip"])
    users = users.set_index("user")[["age", "occupation"]]
    return _encode("ml-100k", df, users, items)


def load_movielens_1m(path: str | Path) -> Dataset:
    """https://grouplens.org/datasets/movielens/1m/ (unzip into data/ml-1m)."""
    path = Path(path)
    df = pd.read_csv(path / "ratings.dat", sep="::", engine="python",
                     names=["user", "item", "rating", "ts"])
    items = pd.read_csv(path / "movies.dat", sep="::", engine="python",
                        names=["item", "title", "genres"], encoding="latin-1")
    items["year"] = items["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    items["title"] = items["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    items["genres"] = items["genres"].str.split("|")
    users = pd.read_csv(path / "users.dat", sep="::", engine="python",
                        names=["user", "gender", "age", "occupation", "zip"])
    users["occupation"] = users["occupation"].astype(str)
    return _encode("ml-1m", df, users.set_index("user")[["age", "occupation"]],
                   items.set_index("item")[["title", "year", "genres"]])


def _read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_amazon_reviews(reviews_path: str | Path, meta_path: str | Path | None = None,
                        min_user: int = 5, min_item: int = 5) -> Dataset:
    """Amazon Reviews (McAuley lab) jsonl / jsonl.gz.

    Handles both the 2023 schema (user_id, parent_asin, rating, timestamp[ms]) and the
    2018 schema (reviewerID, asin, overall, unixReviewTime[s]). Applies the usual
    k-core filter because raw Amazon data is extremely sparse.
    """
    rows = []
    for r in _read_jsonl(Path(reviews_path)):
        user = r.get("user_id", r.get("reviewerID"))
        item = r.get("parent_asin", r.get("asin"))
        rating = r.get("rating", r.get("overall", 1.0))
        ts = r.get("timestamp")
        ts = int(ts) // 1000 if ts is not None else int(r.get("unixReviewTime", 0))
        rows.append((user, item, rating, ts))
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    for _ in range(10):  # iterate k-core to a fixed point
        before = len(df)
        df = df[df.groupby("item")["user"].transform("size") >= min_item]
        df = df[df.groupby("user")["item"].transform("size") >= min_user]
        if len(df) == before:
            break
    items = None
    if meta_path:
        keep = set(df["item"].unique())
        meta = []
        for m in _read_jsonl(Path(meta_path)):
            iid = m.get("parent_asin", m.get("asin"))
            if iid in keep:
                cats = m.get("categories") or m.get("category") or []
                cats = [c for c in cats if isinstance(c, str)]
                meta.append((iid, m.get("title", ""), np.nan, cats))
        items = pd.DataFrame(meta, columns=["item", "title", "year", "genres"]).drop_duplicates("item").set_index("item")
    name = re.sub(r"\.jsonl?(\.gz)?$", "", Path(reviews_path).name)
    return _encode(f"amazon-{name}", df, None, items)


def load(name: str, root: str | Path = "data") -> Dataset:
    root = Path(root)
    if name == "ml-100k":
        return load_movielens_100k(root / "ml-100k")
    if name == "ml-1m":
        return load_movielens_1m(root / "ml-1m")
    if name.startswith("amazon:"):
        # amazon:<reviews file>[,<meta file>]
        parts = name.split(":", 1)[1].split(",")
        return load_amazon_reviews(root / parts[0], root / parts[1] if len(parts) > 1 else None)
    raise ValueError(f"unknown dataset {name!r}")


# --------------------------------------------------------------------------- splits
@dataclass
class Stage:
    """Everything a model may see at one stage, plus the held-out targets."""
    name: str
    history: pd.DataFrame          # user, item, rating, ts — chronologically sorted
    targets: np.ndarray            # (n_users,) target item per user
    n_users: int
    n_items: int
    item_meta: pd.DataFrame
    user_meta: pd.DataFrame
    _cache: dict = field(default_factory=dict, repr=False)

    def matrix(self) -> sp.csr_matrix:
        """Binary user x item matrix of the visible history."""
        if "X" not in self._cache:
            h = self.history
            X = sp.csr_matrix((np.ones(len(h), dtype=np.float32), (h["user"], h["item"])),
                              shape=(self.n_users, self.n_items))
            X.data[:] = 1.0
            self._cache["X"] = X
        return self._cache["X"]

    def sequences(self) -> list[np.ndarray]:
        """Chronological item sequence per user."""
        if "seq" not in self._cache:
            seqs = [np.empty(0, dtype=np.int64)] * self.n_users
            for u, g in self.history.groupby("user", sort=False):
                seqs[u] = g["item"].to_numpy()
            self._cache["seq"] = seqs
        return self._cache["seq"]

    def item_popularity(self) -> np.ndarray:
        return np.asarray(self.matrix().sum(axis=0)).ravel()

    def history_lengths(self) -> np.ndarray:
        return np.asarray(self.matrix().sum(axis=1)).ravel()


@dataclass
class Split:
    tune: Stage   # stage A: history -> val
    test: Stage   # stage B: history + val -> test
    cold_users: np.ndarray  # bool mask over users
    _df: pd.DataFrame = field(repr=False, default=None)
    _meta: tuple = field(repr=False, default=None)

    def stage_at(self, offset: int, name: str | None = None) -> Stage:
        """Stage whose target is each user's `offset`-th-from-last interaction and whose
        history is everything before it. offset=0 is test, offset=1 is tune; offset>=2
        gives earlier slices used only to give the ranker more training queries."""
        df, (n_users, n_items, im, um) = self._df, self._meta
        pos = df["_pos"]
        rows = df[pos == offset]
        t = np.full(n_users, -1, dtype=np.int64)
        t[rows["user"].to_numpy()] = rows["item"].to_numpy()
        hist = df[pos > offset].drop(columns=["_pos", "_order"]).reset_index(drop=True)
        return Stage(name or f"slice{offset}", hist, t, n_users, n_items, im, um)


def leave_last_out(ds: Dataset, cold_frac: float = 0.2, cold_keep: int = 5,
                   seed: int = 0) -> Split:
    """Chronological leave-last-out with a simulated cold-start cohort.

    MovieLens guarantees >= 20 ratings per user, so there are no natural cold users.
    We pick `cold_frac` of users at random and keep only their last `cold_keep + 2`
    interactions (cold_keep for history, 1 val, 1 test). At test time those users
    have cold_keep + 1 known interactions. Everything else about them is untouched.
    """
    df = ds.interactions.copy()
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["user", "ts", "_order"], kind="stable")
    rng = np.random.default_rng(seed)
    cold = np.zeros(ds.n_users, dtype=bool)
    counts = df.groupby("user").size()
    eligible = counts.index[counts >= cold_keep + 2 + 3].to_numpy()  # must actually lose data
    n_cold = int(round(cold_frac * len(eligible)))
    cold[rng.choice(eligible, size=n_cold, replace=False)] = True

    pos_from_end = df.groupby("user").cumcount(ascending=False)  # 0 = last
    keep = ~cold[df["user"].to_numpy()] | (pos_from_end < cold_keep + 2).to_numpy()
    df = df[keep]
    df = df[df.groupby("user")["item"].transform("size") >= 3].copy()  # need history+val+test
    df["_pos"] = df.groupby("user").cumcount(ascending=False)

    split = Split(None, None, cold, df, (ds.n_users, ds.n_items, ds.item_meta, ds.user_meta))
    split.tune, split.test = split.stage_at(1, "tune"), split.stage_at(0, "test")
    return split
