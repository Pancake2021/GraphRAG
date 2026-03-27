from src.extraction.relations import parse_relations


def test_parse_relations_filters_invalid_predicate():
    payload = {
        "relations": [
            {"subject": "Ivan", "predicate": "SOLVED", "object": "Auth bug", "confidence": 0.9},
            {"subject": "Maria", "predicate": "UNKNOWN", "object": "Idea", "confidence": 0.2},
        ]
    }
    parsed = parse_relations(payload, source_chunk_id="c1")
    assert len(parsed) == 1
    assert parsed[0].predicate == "SOLVED"
    assert parsed[0].source_chunk_id == "c1"


def test_parse_relations_accepts_russian_predicates():
    payload = {
        "relations": [
            {"subject": "Иван", "predicate": "РЕШИЛ", "object": "Проблема auth", "confidence": 0.9},
            {"subject": "Мария", "predicate": "ПРЕДЛОЖИЛА", "object": "База знаний", "confidence": 0.8},
        ]
    }
    parsed = parse_relations(payload, source_chunk_id="c2")
    assert len(parsed) == 2
    assert parsed[0].predicate == "SOLVED"
    assert parsed[1].predicate == "PROPOSED"
