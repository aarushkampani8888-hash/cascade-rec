import numpy as np
import pandas as pd

from hrec.ranker import LambdaRanker, LinearBlend

RET = ["pop", "itemknn", "ease", "content", "lightgcn", "sasrec"]


def fake_features(n_q=120, n_c=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        pos = rng.integers(n_c)
        for c in range(n_c):
            lab = int(c == pos)
            row = {"qid": q, "slice": q % 2, "user": q % 60, "item": c, "label": lab}
            for r in RET:
                z = rng.normal() + (1.5 * lab if r in ("sasrec", "lightgcn") else 0)
                row[f"{r}__z"], row[f"{r}__score"], row[f"{r}__logrank"] = z, 3 * z, np.log1p(abs(z))
            row["user__loghist"] = np.log1p(q % 7)
            rows.append(row)
    return pd.DataFrame(rows)


def test_contributions_reproduce_scores():
    df = fake_features()
    rk = LambdaRanker(init_blend=True, num_leaves=7, min_data_in_leaf=20).fit(df)
    c = rk.contributions(df)
    assert np.allclose(c.sum(axis=1).to_numpy(), rk.predict(df), atol=1e-6)


def test_blend_finds_informative_retrievers():
    df = fake_features()
    w = LinearBlend(RET).fit(df).weights()
    assert w["sasrec"] + w["lightgcn"] > 0.6
