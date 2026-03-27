from __future__ import annotations

import re

from src.models import ChunkRecord, DocumentRecord


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def documents_to_chunks(
    docs: list[DocumentRecord], chunk_size: int = 800, overlap: int = 120
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                ChunkRecord(
                    id=f"{doc.id}_c{idx}",
                    doc_id=doc.id,
                    text=chunk,
                    metadata={"doc_type": doc.doc_type, **doc.metadata},
                )
            )
    return chunks
