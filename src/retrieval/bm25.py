from __future__ import annotations

import re
from typing import Callable

from rank_bm25 import BM25Okapi

from src.models import ChunkRecord

_RU_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "о",
    "об",
    "что",
    "это",
    "как",
    "для",
    "к",
    "из",
    "за",
    "у",
    "не",
    "но",
    "или",
    "а",
}
_EN_STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are"}


def _try_get_lemmatizer() -> Callable[[str], str] | None:
    try:
        import pymorphy3  # type: ignore

        morph = pymorphy3.MorphAnalyzer()
        return lambda token: morph.parse(token)[0].normal_form
    except Exception:
        return None


_LEMMATIZE = _try_get_lemmatizer()


def _normalize_token(token: str) -> str:
    token = token.lower().replace("ё", "е")
    if _LEMMATIZE and re.search(r"[а-я]", token):
        return _LEMMATIZE(token)
    return token


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    norm = [_normalize_token(t) for t in raw if len(t) > 1]
    return [t for t in norm if t not in _RU_STOPWORDS and t not in _EN_STOPWORDS]


class BM25Retriever:
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks
        self._tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int = 10) -> list[tuple[ChunkRecord, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idx_and_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i], float(score)) for i, score in idx_and_scores]
