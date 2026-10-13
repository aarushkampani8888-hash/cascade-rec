"""Run the full experiment and write results/ (metrics.json, RESULTS.md, figures).

    python scripts/run_experiment.py                 # ML-100K, ~35 min on 2 CPU cores the first time; reruns reuse results/cache.pkl
    python scripts/run_experiment.py --quick         # smoke run, ~1 min, numbers meaningless
    python scripts/run_experiment.py --dataset ml-1m # after unzipping ml-1m into data/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrec.pipeline import Config, run  # noqa: E402
from hrec.report import write_report  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="ml-100k")
p.add_argument("--data-root", default="data")
p.add_argument("--out", default="results")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--quick", action="store_true", help="few epochs, no ablations")
p.add_argument("--no-ablations", action="store_true")
p.add_argument("--rebuild", action="store_true", help="ignore cached retrieval/features and refit everything")
a = p.parse_args()

cfg = Config(dataset=a.dataset, data_root=a.data_root, out=a.out, seed=a.seed,
             ablations=not (a.quick or a.no_ablations))
if a.quick:
    cfg.sasrec_max_epochs, cfg.lightgcn_max_epochs = 10, 10
cache = Path(a.out) / "cache.pkl"
if a.rebuild and cache.exists():
    cache.unlink()
R = run(cfg, cache_path=cache)
write_report(R, Path(a.out))
