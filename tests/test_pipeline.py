import json

from conftest import synthetic
from hrec.pipeline import Config, run
from hrec.report import write_report


def test_end_to_end_smoke(tmp_path):
    ds = synthetic(n_users=60, n_items=40)
    cfg = Config(out=str(tmp_path), sasrec_max_epochs=5, lightgcn_max_epochs=5, ablations=False,
                 ranker_slices=2, n_candidates={"ease": 15, "sasrec": 15, "lightgcn": 15,
                                                "itemknn": 10, "content": 10, "pop": 5})
    R = run(cfg, ds=ds)
    write_report(R, tmp_path)
    assert (tmp_path / "RESULTS.md").exists()
    names = {r["model"] for r in R["results"]}
    assert {"hybrid (learned ranker)", "ease", "sasrec", "lightgcn"} <= names
    assert 0 <= R["candidates"]["recall_union"] <= 1
    json.loads((tmp_path / "metrics.json").read_text())
