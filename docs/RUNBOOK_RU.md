# Runbook: полный запуск GraphRAG MVP+

## 1) Подготовка
```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
cp .env.example .env
```

Если `uv` не установлен, можно использовать `python -m venv .venv` и `pip install -e '.[dev]'`.

## 2) Поднять Ollama и модель
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

## 3) Проверка окружения
```bash
python -m src.pipeline preflight
```

## 4) Full ingest и индексация
```bash
python -m src.pipeline ingest-full --n 30 --lang ru
```

## 5) Проверка качества extraction (golden-set)
```bash
python -m src.pipeline quality-check --golden data/golden/golden_relations.jsonl --lang ru
```

## 6) Benchmark baseline vs GraphRAG
```bash
python -m src.pipeline benchmark --cases data/benchmark/cases.jsonl --top-k 5 --lang ru
```

## 7) Полный сквозной прогон одной командой
```bash
python -m src.pipeline full-run \
  --n 30 \
  --question "Кто предложил решение проблемы с JWT?" \
  --top-k 5 \
  --questions data/eval/questions.json \
  --benchmark-cases data/benchmark/cases.jsonl \
  --lang ru
```

## 8) Где смотреть результаты
- `data/raw/documents.jsonl`
- `data/ingest/manifest.json`
- `data/processed/chunks.jsonl`
- `data/processed/graph.gpickle`
- `data/processed/evaluation.json`
- `data/processed/quality_report.json`
- `data/processed/benchmark_report.json`
- `data/processed/interview_report.json`
- `data/ops/metrics.json`
- `data/ops/metrics.prom`
- `data/ops/alerts.json`

## 9) Health и retention
```bash
python -m src.pipeline health --mode ready
python -m src.pipeline retention-check --profile stage
```

## Частые проблемы
- `model ... is not pulled`: выполнить `ollama pull qwen2.5:7b-instruct`.
- `ReadTimeout`: увеличить `OLLAMA_TIMEOUT` и уменьшить `SYNTHETIC_BATCH_SIZE` в `.env`.
- `zsh: no matches found: .[dev]`: использовать кавычки `'.[dev]'`.

## Совместимость с EN
- Внутренняя схема графа остаётся EN (`PROPOSED`, `SOLVED`, ...).
- Для явного EN-режима: `--lang en`.
