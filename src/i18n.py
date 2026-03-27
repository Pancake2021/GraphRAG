from __future__ import annotations

import re
from typing import Literal

Lang = Literal["ru", "en"]

SUPPORTED_LANGUAGES: set[str] = {"ru", "en"}

NODE_TYPE_LABELS: dict[str, dict[str, str]] = {
    "ru": {
        "Person": "Персона",
        "Problem": "Проблема",
        "Solution": "Решение",
        "Decision": "Решение команды",
        "Meeting": "Встреча",
        "Document": "Документ",
        "Organization": "Организация",
        "Date": "Дата",
    },
    "en": {
        "Person": "Person",
        "Problem": "Problem",
        "Solution": "Solution",
        "Decision": "Decision",
        "Meeting": "Meeting",
        "Document": "Document",
        "Organization": "Organization",
        "Date": "Date",
    },
}

PREDICATE_LABELS: dict[str, dict[str, str]] = {
    "ru": {
        "PROPOSED": "ПРЕДЛОЖИЛ",
        "SOLVED": "РЕШИЛ",
        "PARTICIPATED_IN": "УЧАСТВОВАЛ В",
        "RELATED_TO": "СВЯЗАНО С",
        "MENTIONED_IN": "УПОМЯНУТО В",
    },
    "en": {
        "PROPOSED": "PROPOSED",
        "SOLVED": "SOLVED",
        "PARTICIPATED_IN": "PARTICIPATED_IN",
        "RELATED_TO": "RELATED_TO",
        "MENTIONED_IN": "MENTIONED_IN",
    },
}

RUSSIAN_TO_EN_PREDICATE: dict[str, str] = {
    "ПРЕДЛОЖИЛ": "PROPOSED",
    "ПРЕДЛОЖИЛА": "PROPOSED",
    "ПРЕДЛОЖЕНО": "PROPOSED",
    "РЕШИЛ": "SOLVED",
    "РЕШИЛА": "SOLVED",
    "РЕШЕНО": "SOLVED",
    "УЧАСТВОВАЛ В": "PARTICIPATED_IN",
    "УЧАСТВОВАЛ": "PARTICIPATED_IN",
    "СВЯЗАНО С": "RELATED_TO",
    "СВЯЗАН": "RELATED_TO",
    "УПОМЯНУТО В": "MENTIONED_IN",
    "УПОМИНАЛ": "MENTIONED_IN",
}


def detect_language(text: str | None) -> Lang:
    if not text:
        return "ru"
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyr >= lat else "en"


def resolve_language(
    preferred: str | None,
    text_hint: str | None,
    default_language: str,
    supported_languages: list[str],
) -> Lang:
    supported = set(supported_languages) & SUPPORTED_LANGUAGES
    if not supported:
        supported = {"ru", "en"}

    if preferred and preferred in supported:
        return preferred  # type: ignore[return-value]

    detected = detect_language(text_hint)
    if detected in supported:
        return detected

    if default_language in supported:
        return default_language  # type: ignore[return-value]
    return "ru"


def normalize_predicate(raw: str) -> str:
    value = raw.strip().upper()
    if value in {"PROPOSED", "SOLVED", "PARTICIPATED_IN", "RELATED_TO", "MENTIONED_IN"}:
        return value
    return RUSSIAN_TO_EN_PREDICATE.get(value, value)


def node_type_label(node_type: str, lang: Lang) -> str:
    return NODE_TYPE_LABELS.get(lang, NODE_TYPE_LABELS["ru"]).get(node_type, node_type)


def predicate_label(predicate: str, lang: Lang) -> str:
    return PREDICATE_LABELS.get(lang, PREDICATE_LABELS["ru"]).get(predicate, predicate)


def synthetic_prompt(lang: Lang, count: int) -> str:
    if lang == "ru":
        return f"""Сгенерируй синтетические документы командной коммуникации для вымышленного ИТ-проекта.

Требования:
- Верни валидный JSON с ключом: documents
- documents — список объектов с полями:
  - doc_type: одно из chat, meeting, brainstorm
  - text: реалистичный русский текст 80-220 слов
  - metadata: JSON-объект (например date, attendees, topic)
- Должны быть люди, проблемы, решения, договоренности
- Количество: {count}
"""
    return f"""Generate synthetic team communication documents for a fictional IT project.

Requirements:
- Return valid JSON object with key: documents
- documents must be a list of objects with keys:
  - doc_type: one of chat, meeting, brainstorm
  - text: realistic text, 80-220 words each
  - metadata: JSON object with optional keys like date, attendees, topic
- Include people, problems, solutions, and decisions
- Count: {count}
"""


def relations_prompt(lang: Lang, text: str) -> str:
    if lang == "ru":
        return f"""Извлеки отношения из текста.

Верни JSON-объект с ключом `relations`, где каждый элемент имеет:
- subject (строка)
- predicate (одно из PROPOSED, SOLVED, PARTICIPATED_IN, RELATED_TO, MENTIONED_IN)
- object (строка)
- confidence (0..1)

Текст:
{text}
"""
    return f"""Extract relations from the text.

Return JSON object with key `relations`, where each relation has:
- subject (string)
- predicate (one of PROPOSED, SOLVED, PARTICIPATED_IN, RELATED_TO, MENTIONED_IN)
- object (string)
- confidence (0..1)

Text:
{text}
"""


def answer_prompt(lang: Lang, query: str, graph_ctx_json: str, chunks_text: str) -> str:
    if lang == "ru":
        return (
            "Ответь на вопрос только по контексту. "
            "Дай краткий ответ и явно укажи источники (ID чанков).\n"
            f"Вопрос: {query}\n\n"
            f"Контекст графа: {graph_ctx_json}\n\n"
            f"Чанки:\n{chunks_text}\n\n"
            "Верни обычный текст."
        )
    return (
        "Answer the question using only the context. "
        "Provide concise answer and explicit source chunk IDs.\n"
        f"Question: {query}\n\n"
        f"Graph context: {graph_ctx_json}\n\n"
        f"Chunks:\n{chunks_text}\n\n"
        "Return plain text answer."
    )

