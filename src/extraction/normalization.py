from __future__ import annotations

import hashlib
import re


_ROLE_WORDS = {
    "разработчик",
    "менеджер",
    "руководитель",
    "аналитик",
    "инженер",
    "developer",
    "manager",
    "lead",
}


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_entity_id(entity_type: str, value: str) -> str:
    base = _norm(value)
    digest = hashlib.sha1(f"{entity_type}:{base}".encode("utf-8")).hexdigest()[:12]
    return f"{entity_type.lower()}_{digest}"


def normalize_entities(entities: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, str], int]:
    normalized: dict[str, list[str]] = {k: [] for k in entities.keys()}
    mapping: dict[str, str] = {}
    merged = 0

    for etype, values in entities.items():
        seen_norm: dict[str, str] = {}
        for raw in values:
            norm = _norm(raw)
            if not norm:
                continue
            display = " ".join(x.capitalize() for x in norm.split()) if etype == "Person" else raw.strip()
            if norm in seen_norm:
                merged += 1
                mapping[raw] = seen_norm[norm]
                continue
            seen_norm[norm] = display
            mapping[raw] = display
            normalized[etype].append(display)

    if "Person" in normalized:
        role_candidates = [x for x in normalized["Person"] if _norm(x) in _ROLE_WORDS]
        if role_candidates:
            normalized.setdefault("Role", [])
            normalized["Role"].extend(role_candidates)

    for k in list(normalized.keys()):
        normalized[k] = sorted(set(normalized[k]))
    return normalized, mapping, merged
