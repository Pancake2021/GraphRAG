from __future__ import annotations

from dataclasses import dataclass

from src.models import Relation


_ALLOWED_TYPES = {
    "PROPOSED": {("Person", "Solution"), ("Person", "Decision")},
    "SOLVED": {("Person", "Problem")},
    "PARTICIPATED_IN": {("Person", "Meeting")},
    "RELATED_TO": {("Problem", "Solution"), ("Decision", "Problem"), ("Decision", "Solution")},
    "MENTIONED_IN": {("Person", "Document"), ("Problem", "Document"), ("Solution", "Document"), ("Decision", "Document")},
}


@dataclass
class RelationValidationResult:
    valid: list[Relation]
    invalid: list[dict]


def _guess_type(value: str) -> str:
    low = value.lower()
    if any(x in low for x in ["встреч", "митинг", "meeting", "sprint"]):
        return "Meeting"
    if any(x in low for x in ["проблем", "ошиб", "bug", "issue", "deadline"]):
        return "Problem"
    if any(x in low for x in ["решени", "фик", "solution", "upgrade", "предлож"]):
        return "Solution"
    words = value.split()
    if len(words) >= 2 and all(w[:1].isupper() for w in words if w):
        return "Person"
    return "Decision"


def validate_relations(relations: list[Relation], min_confidence: float = 0.5) -> RelationValidationResult:
    valid: list[Relation] = []
    invalid: list[dict] = []

    for rel in relations:
        if rel.confidence < min_confidence:
            invalid.append({"relation": rel.model_dump(), "reason": "low_confidence"})
            continue

        s_type = _guess_type(rel.subject)
        o_type = _guess_type(rel.object)
        allowed = _ALLOWED_TYPES.get(rel.predicate, set())
        if allowed and (s_type, o_type) not in allowed:
            invalid.append(
                {
                    "relation": rel.model_dump(),
                    "reason": "schema_violation",
                    "subject_type": s_type,
                    "object_type": o_type,
                }
            )
            continue
        valid.append(rel)

    return RelationValidationResult(valid=valid, invalid=invalid)
