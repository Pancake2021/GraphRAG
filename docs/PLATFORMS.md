# Платформенная совместимость: Mac M1 и Ryzen + RTX 4060 8GB

## Поддерживаемые ОС и CPU
- macOS (Apple Silicon M1/M2/M3): поддерживается.
- Linux x86_64 (Ryzen): поддерживается.
- Windows: рекомендуется запуск через WSL2 Ubuntu для максимальной совместимости Python/NLP стеков.

## Ollama и ускорение
- Mac M1:
  - Ollama использует Metal backend автоматически.
  - Обычно достаточно `qwen2.5:7b-instruct` в 4/8-bit квантизации.

- Ryzen + RTX 4060 8GB:
  - Для Linux/WSL2 рекомендуется NVIDIA driver + CUDA runtime.
  - Ollama может работать через GPU при корректной установке драйверов.
  - Для 8GB VRAM оптимально использовать 7B-класс модели в квантизированном виде.

## Рекомендованные профили
- Базовый (универсальный):
  - `OLLAMA_MODEL=qwen2.5:7b-instruct`
  - `CHUNK_SIZE=800`
  - `CHUNK_OVERLAP=120`

- Память/скорость (если не хватает ресурсов):
  - уменьшить `CHUNK_SIZE` до `500-650`;
  - уменьшить `TOP_K` до `3-4`;
  - уменьшить количество документов в `full-run` (`--n 10..20`).

## Проверка на двух машинах
На каждой машине выполнить:
1. `python -m src.pipeline preflight`
2. `python -m src.pipeline full-run --n 10 --top-k 3 --questions data/eval/questions.json`
3. Проверить, что создан `data/processed/evaluation.json` и нет ошибок в консоли.
