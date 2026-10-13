"""Show a user's top recommendations with reasons (reads results/explanations.pkl.gz).

    python scripts/recommend.py --user 196
    python scripts/recommend.py --random-cold
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrec.explain import explain  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--user", help="raw user id from the dataset")
p.add_argument("--random-cold", action="store_true")
p.add_argument("--k", type=int, default=10)
p.add_argument("--results", default="results")
a = p.parse_args()

art = pd.read_pickle(Path(a.results) / "explanations.pkl.gz")
top, gc, ctx, raw = art["top"], art["group_contrib"], art["ctx"], art["raw_user_ids"]
raw_to_int = {str(r): i for i, r in enumerate(raw)}
if a.random_cold:
    u = int(np.random.default_rng().choice(np.flatnonzero(art["cold_users"])))
else:
    u = raw_to_int[str(a.user or raw[0])]

meta, seq = ctx["item_meta"], ctx["sequences"][u]
print(f"\nUser {raw[u]}  ({'cold-start, ' if art['cold_users'][u] else ''}{len(seq)} known interactions)")
print("Recently watched:", " · ".join(meta.at[i, "title"] for i in seq[-6:]))
rows = top[top["user"] == u].sort_values("rank").head(a.k)
print()
for i, rec in enumerate(explain(rows, gc, ctx, peers=top[top["user"] == u]), 1):
    print(f"{i:2d}. {rec['title']}")
    for reason in rec["reasons"]:
        print(f"      - {reason}")
    drivers = ", ".join(f"{k} {v:+.2f}" for k, v in rec["drivers"].items())
    print(f"      SHAP: {drivers}")
