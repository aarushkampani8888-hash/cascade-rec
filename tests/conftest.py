import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrec.data import _encode  # noqa: E402


def synthetic(n_users=80, n_items=50, per_user=(12, 25), seed=0):
    """Two taste clusters + a sequential habit (users tend to go to item+1 next)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        cluster = u % 2
        pool = np.arange(n_items)[(np.arange(n_items) % 2) == cluster]
        n = rng.integers(*per_user)
        cur = rng.choice(pool)
        for t in range(n):
            rows.append((f"u{u}", f"i{cur}", float(rng.integers(1, 6)), 1_000_000 + 3600 * (t + 50 * u)))
            cur = (cur + 2) % n_items if rng.random() < 0.7 else rng.choice(pool)
    df = pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])
    items = pd.DataFrame({"item": [f"i{i}" for i in range(n_items)],
                          "title": [f"Movie {i} {'space' if i % 2 else 'love'}" for i in range(n_items)],
                          "year": 1990 + np.arange(n_items) % 10,
                          "genres": [["Sci-Fi"] if i % 2 else ["Romance"] for i in range(n_items)]}).set_index("item")
    users = pd.DataFrame({"user": [f"u{u}" for u in range(n_users)], "age": 20 + np.arange(n_users) % 40,
                          "occupation": ["student", "engineer"] * (n_users // 2)}).set_index("user")
    return _encode("synthetic", df, users, items)


@pytest.fixture
def ds():
    return synthetic()
