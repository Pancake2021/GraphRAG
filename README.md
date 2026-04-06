# GraphRAG Platform

CLI-first платформа для анализа рабочих коммуникаций через GraphRAG.

## Что внутри
- NER/relations extraction (Natasha + LLM)
- Knowledge Graph (NetworkX)
- Hybrid retrieval (BM25 + vector + graph expansion)
- Ответы с источниками
- Data governance (PII, versioning, retention)
- Quality/benchmark/evaluation (включая RAGAS)

## Быстрый старт
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

```bash
python -m src.pipeline preflight
```

## Полный запуск
```bash
python -m src.pipeline full-run \
  --n 30 \
  --question "Кто предложил решение проблемы с JWT?" \
  --top-k 5 \
  --questions data/eval/questions.json \
  --benchmark-cases data/benchmark/cases.jsonl \
  --lang ru
```

## Основные CLI команды
```bash
python -m src.pipeline ingest-full --n 30 --lang ru
python -m src.pipeline ingest-incremental --lang ru
python -m src.pipeline query "Кто предложил исправление по JWT?" --top-k 5 --lang ru
python -m src.pipeline evaluate --questions data/eval/questions.json --top-k 5 --lang ru
python -m src.pipeline evaluate-ragas --questions data/eval/questions.json --top-k 5 --lang ru
python -m src.pipeline quality-check --golden data/golden/golden_relations.jsonl --lang ru
python -m src.pipeline benchmark --cases data/benchmark/cases.jsonl --top-k 5 --lang ru
python -m src.pipeline health --mode ready
```

## Артефакты
- `data/processed/graph.gpickle`, `data/processed/graph.json`
- `data/processed/evaluation.json`
- `data/processed/ragas_evaluation.json`
- `data/processed/quality_report.json`
- `data/processed/benchmark_report.json`
- `data/processed/interview_report.json`
- `data/ops/metrics.json`, `data/ops/metrics.prom`, `data/ops/alerts.json`

## Документация
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK_RU.md`
- `docs/CLI_GUIDE_RU.md`
- `docs/PLATFORMS.md`
