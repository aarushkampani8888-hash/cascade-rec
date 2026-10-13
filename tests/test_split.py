import numpy as np

from hrec.data import leave_last_out


def test_no_leakage_and_chronology(ds):
    s = leave_last_out(ds, cold_frac=0.25, cold_keep=5, seed=1)
    A, B = s.tune, s.test
    users = np.where(B.targets >= 0)[0]
    assert len(users) > 0
    for u in users:
        hA = A.history[A.history.user == u]
        hB = B.history[B.history.user == u]
        val, test = A.targets[u], B.targets[u]
        assert val not in set(hA.item) and test not in set(hA.item)  # val/test never in tune history
        assert test not in set(hB.item)                              # test never in test history
        assert set(hB.item) == set(hA.item) | {val}                  # test history = tune history + val
        ts = ds.interactions
        t_val = ts[(ts.user == u) & (ts.item == val)].ts.iloc[0]
        t_test = ts[(ts.user == u) & (ts.item == test)].ts.iloc[0]
        assert hA.ts.max() <= t_val <= t_test


def test_cold_cohort_truncated(ds):
    s = leave_last_out(ds, cold_frac=0.25, cold_keep=5, seed=1)
    hl = s.tune.history_lengths()
    assert s.cold_users.sum() > 0
    assert np.all(hl[s.cold_users] == 5)
    raw = ds.interactions.groupby("user").size().reindex(range(ds.n_users), fill_value=0).to_numpy()
    warm = ~s.cold_users & (s.tune.targets >= 0)
    assert np.all(hl[warm] == raw[warm] - 2)  # warm users lose only val + test


def test_earlier_slices_are_strictly_earlier(ds):
    s = leave_last_out(ds, seed=0)
    s2 = s.stage_at(2)
    assert len(s2.history) < len(s.tune.history) < len(s.test.history)
    for u in np.where(s2.targets >= 0)[0][:20]:
        assert s2.targets[u] not in set(s2.history[s2.history.user == u].item)
