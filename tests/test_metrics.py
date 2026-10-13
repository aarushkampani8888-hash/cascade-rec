import numpy as np
import scipy.sparse as sp

from hrec.metrics import mask_seen, paired_bootstrap, per_user_metrics, target_ranks


def test_ranks_and_ndcg():
    s = np.array([[0.9, 0.5, 0.1], [0.2, 0.2, 0.9]])
    r = target_ranks(s, np.array([0, 1]))
    assert list(r) == [0, 2]          # row 2: one item above, one tie -> tie counts against target
    m = per_user_metrics(r, ks=(1,))
    assert m["NDCG@1"][0] == 1.0 and m["NDCG@1"][1] == 0.0
    assert np.isclose(m["MRR"][1], 1 / 3)


def test_constant_scores_rank_last():
    s = np.zeros((1, 100))
    assert target_ranks(s, np.array([42]))[0] == 99


def test_mask_seen_and_out_of_candidates():
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0]]))
    s = mask_seen(np.array([[5.0, 1.0, 2.0, 3.0]]), seen, np.array([0]))
    assert np.isneginf(s[0, 0])
    s[0, 1] = -np.inf  # target outside candidate set
    assert target_ranks(s, np.array([1]))[0] == 4


def test_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    b = rng.random(500)
    out = paired_bootstrap(b + 0.05, b)
    assert out["ci_low"] > 0
