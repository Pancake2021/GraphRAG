from src.extraction.ner import extract_entities, normalize_entity


def test_normalize_entity():
    assert normalize_entity("  Ivan   Petrov ") == "Ivan Petrov"


def test_extract_entities_basic():
    text = "Ivan Petrov met Maria Sidorova on 2025-01-15 in HR Office"
    entities = extract_entities(text)
    assert "Ivan Petrov" in entities["Person"]
    assert "Maria Sidorova" in entities["Person"]
    assert "2025-01-15" in entities["Date"]
