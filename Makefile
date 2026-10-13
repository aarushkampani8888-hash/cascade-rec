.PHONY: install test quick experiment bench demo

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

quick:            ## ~3 min smoke run; numbers are not meaningful
	python scripts/run_experiment.py --quick --out results/quick

experiment:       ## full run on ML-100K (~30-40 min on a 2-core CPU, much faster on a laptop)
	python scripts/run_experiment.py

bench:            ## exact vs HNSW retrieval
	python scripts/bench_ann.py

demo:
	python scripts/recommend.py --user 1
