.PHONY: setup preflight full-run ingest-full ingest-incremental quality-check benchmark retention-check health-ready test clean

setup:
	uv venv
	. .venv/bin/activate && uv pip install -e '.[dev]'

preflight:
	python -m src.pipeline preflight

full-run:
	python -m src.pipeline full-run --n 30 --top-k 5 --questions data/eval/questions.json --benchmark-cases data/benchmark/cases.jsonl --lang ru

ingest-full:
	python -m src.pipeline ingest-full --n 30 --lang ru

ingest-incremental:
	python -m src.pipeline ingest-incremental --lang ru

quality-check:
	python -m src.pipeline quality-check --golden data/golden/golden_relations.jsonl --lang ru

benchmark:
	python -m src.pipeline benchmark --cases data/benchmark/cases.jsonl --top-k 5 --lang ru

retention-check:
	python -m src.pipeline retention-check --profile dev

health-ready:
	python -m src.pipeline health --mode ready

test:
	python -m pytest -q

clean:
	rm -rf data/raw/* data/processed/* data/ingest/* data/governance/* data/ops/* .pytest_cache
