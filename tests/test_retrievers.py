import numpy as np

from hrec.data import leave_last_out
from hrec.metrics import mask_seen, target_ranks
from hrec.retrievers import EASE, Content, ItemKNN, LightGCN, Popularity, SASRec


def _ndcg(m, st):
    u = np.where(st.targets >= 0)[0]
    r = target_ranks(mask_seen(m.score(u), st.matrix(), u), st.targets[u])
    return np.mean(np.where(r < 10, 1 / np.log2(r + 2), 0))


def test_ease_structure(ds):
    st = leave_last_out(ds).tune
    m = EASE(10).fit(st)
    assert np.allclose(np.diag(m.B), 0)
    assert m.score(np.array([0, 1])).shape == (2, ds.n_items)


def test_collaborative_models_beat_popularity_on_structured_data(ds):
    st = leave_last_out(ds).tune
    pop = _ndcg(Popularity().fit(st), st)
    for m in (EASE(10), ItemKNN(k=20)):
        assert _ndcg(m.fit(st), st) > pop, m.name


def test_neural_models_train_and_record_epochs(ds):
    st = leave_last_out(ds).tune
    for m in (LightGCN(max_epochs=20, eval_every=5, verbose=False),
              SASRec(max_epochs=20, eval_every=5, verbose=False, max_len=20)):
        m.fit(st, st.targets)
        assert m.best_epoch is not None and m.best_epoch > 0
        assert np.isfinite(m.score(np.array([0, 3]))).all()


def test_content_prefers_users_genre(ds):
    st = leave_last_out(ds).tune
    s = Content().fit(st).score(np.arange(10))
    parity = np.array([int(r[1:]) % 2 for r in ds.raw_item_ids])  # synthetic genre = id parity
    for u in range(10):  # each synthetic user only watches one parity
        mine = parity[st.sequences()[u][0]]
        assert s[u, parity == mine].mean() > s[u, parity != mine].mean()
