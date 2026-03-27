from src.extraction.relation_quality import validate_relations
from src.models import Relation


def test_relation_quality_filters_by_schema_and_confidence():
    relations = [
        Relation(
            subject="Иван Петров",
            predicate="SOLVED",
            object="Проблема дедлайна",
            confidence=0.91,
            source_chunk_id="c1",
        ),
        Relation(
            subject="Иван Петров",
            predicate="SOLVED",
            object="Решение с дедупликацией",
            confidence=0.92,
            source_chunk_id="c1",
        ),
        Relation(
            subject="Мария Сидорова",
            predicate="PROPOSED",
            object="Автоматизация выдачи доступов",
            confidence=0.40,
            source_chunk_id="c2",
        ),
    ]

    result = validate_relations(relations, min_confidence=0.5)
    assert len(result.valid) == 1
    assert len(result.invalid) == 2
    reasons = {row["reason"] for row in result.invalid}
    assert "schema_violation" in reasons
    assert "low_confidence" in reasons

