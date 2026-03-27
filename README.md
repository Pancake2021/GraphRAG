# GraphRAG Platform

GraphRAG Platform — CLI-first сервис для поиска и аналитики по неструктурированным коммуникациям и рабочим документам.

Система объединяет:
- extraction (entities + relations),
- knowledge graph (NetworkX),
- hybrid retrieval (BM25 + vector + graph expansion),
- answer generation with sources.

Проект ориентирован на production-style контур: data governance, incremental ingest, quality gates, observability и benchmark-отчеты.

## Core Capabilities
- RU-first workflow с EN-совместимостью.
- Idempotent ingest с дедупликацией, fingerprint/versioning и lineage.
- PII pseudonymization перед индексацией и генерацией.
- Явная schema validation для relations.
- Baseline vs GraphRAG benchmark и quality-check на golden-set.
- Health/metrics/alerts для операционного контроля.

## Tech Stack
- LLM Runtime: Ollama (`qwen2.5:7b-instruct`)
- NLP (NER): Natasha
- Graph: NetworkX
- Vector DB: ChromaDB
- Lexical Retrieval: BM25 (`rank-bm25`)
- CLI/Test: Python, argparse, pytest

## Quick Start
```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
cp .env.example .env
```

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

```bash
python -m src.pipeline preflight
```

## Full Run
```bash
python -m src.pipeline full-run \
  --n 30 \
  --question "Кто предложил решение проблемы с JWT?" \
  --top-k 5 \
  --questions data/eval/questions.json \
  --benchmark-cases data/benchmark/cases.jsonl \
  --lang ru
```

## CLI Interface
```bash
python -m src.pipeline generate-data --n 30 --lang ru
python -m src.pipeline ingest-full --n 30 --lang ru
python -m src.pipeline ingest-incremental --lang ru
python -m src.pipeline build-index
python -m src.pipeline query "Кто предложил исправление по JWT?" --top-k 5 --lang ru
python -m src.pipeline evaluate --questions data/eval/questions.json --top-k 5 --lang ru
python -m src.pipeline quality-check --golden data/golden/golden_relations.jsonl --lang ru
python -m src.pipeline benchmark --cases data/benchmark/cases.jsonl --top-k 5 --lang ru
python -m src.pipeline retention-check --profile stage
python -m src.pipeline health --mode ready
python -m src.pipeline preflight
python -m src.pipeline full-run --n 30 --top-k 5 --questions data/eval/questions.json --benchmark-cases data/benchmark/cases.jsonl --lang ru
```

## Data & Governance Artifacts
- `data/ingest/manifest.json` — ingest state, statuses, version lineage.
- `data/governance/pii_mapping.jsonl` — restricted pseudonymization mapping.
- `data/processed/chunks.jsonl` — processed chunk corpus.
- `data/processed/graph.gpickle` / `graph.json` — knowledge graph.
- `data/processed/quality_report.json` — extraction quality metrics.
- `data/processed/benchmark_report.json` — baseline vs GraphRAG.
- `data/processed/interview_report.json` — consolidated implementation report.
- `data/ops/metrics.json` / `metrics.prom` / `alerts.json` — ops telemetry.

## Performance Notes
Для Mac M1 (стабильный профиль):
```bash
OLLAMA_CONTEXT_LENGTH=2048 OLLAMA_FLASH_ATTENTION=false OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_TIMEOUT=240 SYNTHETIC_BATCH_SIZE=2 CHUNK_SIZE=550 TOP_K=3 python -m src.pipeline full-run --n 20 --question "Кто предложил решение проблемы с JWT?" --top-k 3 --questions data/eval/questions.json --benchmark-cases data/benchmark/cases.jsonl --lang ru
```

## Documentation
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK_RU.md`
- `docs/CLI_GUIDE_RU.md`
- `docs/PLATFORMS.md`

## Notes
- Внутренняя схема графа и предикаты сохраняются в стабильных EN-кодах (`PROPOSED`, `SOLVED`, ...).
- User-facing слой работает в RU-first режиме по умолчанию (`DEFAULT_LANGUAGE=ru`).
