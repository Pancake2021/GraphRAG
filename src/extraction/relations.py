from __future__ import annotations

from src.i18n import Lang, normalize_predicate, relations_prompt
from src.models import Relation

_ALLOWED = {"PROPOSED", "SOLVED", "PARTICIPATED_IN", "RELATED_TO", "MENTIONED_IN"}


def parse_relations(payload: dict, source_chunk_id: str) -> list[Relation]:
    rows = payload.get("relations", [])
    if not isinstance(rows, list):
        return []
    out: list[Relation] = []
    for row in rows:
        pred = normalize_predicate(str(row.get("predicate", "")))
        if pred not in _ALLOWED:
            continue
        subj = str(row.get("subject", "")).strip()
        obj = str(row.get("object", "")).strip()
        if not subj or not obj:
            continue
        conf = float(row.get("confidence", 0.7))
        conf = max(0.0, min(1.0, conf))
        out.append(
            Relation(
                subject=subj,
                predicate=pred,  # type: ignore[arg-type]
                object=obj,
                confidence=conf,
                source_chunk_id=source_chunk_id,
            )
        )
    return out


def extract_relations(llm_client, chunk_id: str, text: str, lang: Lang = "ru") -> list[Relation]:
    system = "Верни только валидный JSON." if lang == "ru" else "You output only valid JSON."
    payload = llm_client.generate_json(
        relations_prompt(lang=lang, text=text),
        system=system,
        temperature=0.0,
    )
    return parse_relations(payload, source_chunk_id=chunk_id)
