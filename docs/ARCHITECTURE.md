# Архитектура GraphRAG MVP+ (Interview-grade)

## Поток данных
1. `generate-data`
- Генерирует синтетические документы через Ollama/Qwen.
- Сохраняет в `data/raw/documents.jsonl`.

2. `ingest-full` / `ingest-incremental`
- Pre-ingest governance:
  - PII-псевдонимизация (`src/governance/pii.py`);
  - dedup/versioning/idempotency (`src/ingest/state.py`);
  - ingest manifest (`data/ingest/manifest.json`).
- Делит документы на чанки (`src/preprocessing/chunker.py`).
- Извлекает сущности (`ner.py`).
- Нормализует сущности (`src/extraction/normalization.py`).
- Извлекает отношения (`src/extraction/relations.py`).
- Валидирует relation schema (`src/extraction/relation_quality.py`).
- Строит граф (`graph/builder.py`) и сохраняет его в `data/processed/graph.gpickle` и `data/processed/graph.json`.
- Индексирует чанки в Chroma (`retrieval/vector_store.py`).
- Сохраняет чанки в `data/processed/chunks.jsonl`.

3. `query`
- Выполняет BM25 + vector retrieval.
- Объединяет результаты через RRF.
- Расширяет контекст через соседей графа.
- Генерирует ответ LLM с источниками.

4. `evaluate`
- Сравнивает baseline (без граф-расширения) и GraphRAG.
- Считает метрики и пишет `data/processed/evaluation.json`.

5. `quality-check`
- Считает precision/recall/F1 relation extraction на golden-set.
- Пишет `data/processed/quality_report.json`.

6. `benchmark`
- Сравнивает baseline vs GraphRAG на benchmark-кейсах.
- Пишет `data/processed/benchmark_report.json` и `data/processed/interview_report.json`.

7. Ops
- Метрики/алерты (`src/ops/metrics_store.py`) -> `data/ops/*`.
- Health checks (`src/ops/health.py`) для `live` и `ready`.
- Retention (`src/governance/retention.py`) через `retention-check`.

## Ключевые модули
- `src/pipeline.py`: оркестрация CLI.
- `src/llm/qwen_client.py`: вызовы Ollama API.
- `src/retrieval/hybrid.py`: RRF + graph expansion.
- `src/graph/*`: схема и обход графа.
- `src/governance/*`: PII и retention.
- `src/ingest/*`: incremental ingest state/versioning.
- `src/ops/*`: метрики, alerts, health.

## Артефакты
- `data/raw/documents.jsonl`
- `data/governance/pii_mapping.jsonl`
- `data/ingest/manifest.json`
- `data/processed/chunks.jsonl`
- `data/processed/graph.gpickle`
- `data/processed/graph.json`
- `data/processed/evaluation.json`
- `data/processed/quality_report.json`
- `data/processed/benchmark_report.json`
- `data/processed/interview_report.json`
- `data/ops/metrics.json`
- `data/ops/metrics.prom`
- `data/ops/alerts.json`
