from __future__ import annotations

import numpy as np

from ..data import Stage


class Retriever:
    """Stage-1 model: sees a Stage's history, returns a score for every (user, item).

    Scores only need to be comparable *within a user*; the ranker gets per-user
    normalized versions, so retrievers are free to use whatever scale is natural.
    """

    name: str = "base"

    def fit(self, stage: Stage) -> "Retriever":
        raise NotImplementedError

    def score(self, users: np.ndarray) -> np.ndarray:
        """Dense (len(users), n_items) score matrix. Seen items are NOT masked here."""
        raise NotImplementedError

    def params(self) -> dict:
        return {}
