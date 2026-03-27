from __future__ import annotations

import logging
import uuid

from src.i18n import Lang, synthetic_prompt
from src.models import DocumentRecord

logger = logging.getLogger(__name__)


def _generate_batch(llm_client, batch_count: int, lang: Lang) -> list[dict]:
    logger.info("Generating synthetic batch: size=%s", batch_count)
    system = (
        "Верни только валидный JSON. Без markdown и пояснений."
        if lang == "ru"
        else "You produce strict JSON only. No markdown, no commentary."
    )

    # Two attempts: normal temperature, then deterministic.
    for temp in (0.4, 0.0):
        data = llm_client.generate_json(
            synthetic_prompt(lang=lang, count=batch_count),
            system=system,
            temperature=temp,
        )
        docs = data.get("documents", [])
        if isinstance(docs, list) and docs:
            return docs
        logger.warning("Synthetic batch returned empty documents. temp=%s size=%s", temp, batch_count)

    raise ValueError("Synthetic generation returned empty 'documents'.")


def generate_synthetic_documents(
    llm_client,
    count: int = 30,
    batch_size: int = 6,
    lang: Lang = "ru",
) -> list[DocumentRecord]:
    batch_size = max(1, batch_size)
    remaining = max(0, count)
    generated: list[dict] = []

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        try:
            generated.extend(_generate_batch(llm_client, current_batch, lang=lang))
        except ValueError:
            if current_batch == 1:
                raise
            logger.warning("Batch generation failed for size=%s. Retrying with size=1.", current_batch)
            generated.extend(_generate_batch(llm_client, 1, lang=lang))
        remaining = count - len(generated)
        logger.info("Synthetic progress: generated=%s target=%s", len(generated), count)

    out: list[DocumentRecord] = []
    for item in generated[:count]:
        doc_type = item.get("doc_type", "generic")
        text = item.get("text", "").strip()
        metadata = item.get("metadata", {}) or {}
        if not text:
            continue
        out.append(
            DocumentRecord(
                id=f"doc_{uuid.uuid4().hex[:12]}",
                doc_type=doc_type if doc_type in {"chat", "meeting", "brainstorm"} else "generic",
                text=text,
                metadata=metadata,
            )
        )
    logger.info("Synthetic generation completed: final_docs=%s", len(out))
    return out
