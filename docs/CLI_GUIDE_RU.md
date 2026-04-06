# Гайд по CLI интерфейсу

## Что такое CLI
`CLI` (Command Line Interface) — интерфейс командной строки: управление проектом через команды в терминале.

Основной интерфейс проекта: `python -m src.pipeline ...`.

## Общий синтаксис
```bash
python -m src.pipeline <команда> [аргументы]
```

## Команды

### 1) `generate-data`
```bash
python -m src.pipeline generate-data --n 30 --lang ru
```
Что делает:
- генерирует синтетические документы через Ollama;
- сохраняет в `data/raw/documents.jsonl`.

Параметры:
- `--n` — число документов;
- `--lang {ru,en}` — язык генерации.

### 2) `ingest-full`
```bash
python -m src.pipeline ingest-full --n 30 --lang ru
```
Что делает:
- запускает `generate-data`;
- применяет pre-ingest governance (PII pseudonymization, versioning);
- строит чанки/relations/граф и индексирует в Chroma.

### 3) `ingest-incremental`
```bash
python -m src.pipeline ingest-incremental --lang ru
```
Что делает:
- читает `data/raw/documents.jsonl`;
- определяет статусы `new/unchanged/updated/invalid`;
- переиндексирует только `new/updated`.

Артефакты:
- `data/ingest/manifest.json`
- `data/ingest/documents_managed.jsonl`
- `data/processed/documents_latest.jsonl`

### 4) `build-index`
```bash
python -m src.pipeline build-index
```
Что делает:
- полная индексация текущих raw-документов;
- строит `chunks.jsonl`, `graph.gpickle`, `graph.json`, обновляет Chroma.

### 5) `query`
```bash
python -m src.pipeline query "Кто предложил решение по JWT?" --top-k 5 --lang ru
```
Что делает:
- hybrid retrieval: BM25 + vector + RRF;
- graph expansion по соседям сущностей;
- LLM-ответ с источниками (`chunk_id`, `doc_id`).

Параметры:
- `query` — текст вопроса;
- `--top-k` — глубина retrieval;
- `--lang {ru,en}` — язык ответа.

### 6) `evaluate`
```bash
python -m src.pipeline evaluate --questions data/eval/questions.json --top-k 5 --lang ru
```
Что делает:
- считает baseline vs GraphRAG на eval-наборе;
- сохраняет отчёт в `data/processed/evaluation.json`.

### 7) `quality-check`
```bash
python -m src.pipeline quality-check --golden data/golden/golden_relations.jsonl --lang ru
```
Что делает:
- прогоняет relation extraction на golden-set;
- считает `precision/recall/F1` и `schema_violations`;
- сохраняет в `data/processed/quality_report.json`.

### 8) `evaluate-ragas`
```bash
python -m src.pipeline evaluate-ragas --questions data/eval/questions.json --top-k 5 --lang ru
```
Что делает:
- прогоняет GraphRAG ответы по eval-набору;
- считает метрики RAGAS:
  - `faithfulness`
  - `context_precision`
  - `context_recall`
- сохраняет отчёт в `data/processed/ragas_evaluation.json`.

### 9) `benchmark`
```bash
python -m src.pipeline benchmark --cases data/benchmark/cases.jsonl --top-k 5 --lang ru
```
Что делает:
- сравнивает baseline vs GraphRAG на benchmark-кейсах;
- считает абсолютный/относительный прирост;
- пишет:
  - `data/processed/benchmark_report.json`
  - `data/processed/interview_report.json`.

### 10) `retention-check`
```bash
python -m src.pipeline retention-check --profile stage
python -m src.pipeline retention-check --profile stage --apply
```
Что делает:
- строит dry-run список кандидатов на удаление по TTL;
- при `--apply` удаляет файлы;
- отчёт: `data/governance/retention_report.json`.

### 11) `health`
```bash
python -m src.pipeline health --mode live
python -m src.pipeline health --mode ready
```
Что делает:
- `live` — liveness процесса;
- `ready` — готовность зависимостей (Ollama/model + обязательные артефакты).

### 12) `preflight`
```bash
python -m src.pipeline preflight
```
Что делает:
- проверяет Python/платформу;
- проверяет `ollama` в PATH;
- проверяет Ollama API и модель.

### 13) `full-run`
```bash
python -m src.pipeline full-run \
  --n 30 \
  --question "Кто решал проблему дедлайна?" \
  --top-k 5 \
  --questions data/eval/questions.json \
  --benchmark-cases data/benchmark/cases.jsonl \
  --lang ru
```
Что делает:
1. `preflight`
2. `ingest-full`
3. `query`
4. `evaluate`
5. `benchmark`
6. алерты в `data/ops/alerts.json`

## Метрики и логи
- Логи запуска: `logs/run-*.log`, `logs/latest.log`.
- Метрики JSON: `data/ops/metrics.json`.
- Prometheus-формат: `data/ops/metrics.prom`.
- Алерты: `data/ops/alerts.json`.

## Makefile (быстрые цели)
```bash
make setup
make preflight
make ingest-full
make ingest-incremental
make quality-check
make ragas-eval
make benchmark
make health-ready
make full-run
make test
```

## Совместимость
- RU-first по умолчанию (`DEFAULT_LANGUAGE=ru`).
- EN поддерживается через `--lang en` и auto-fallback.
- Внутренние коды схемы графа остаются EN (`PROPOSED`, `SOLVED`, ...).
