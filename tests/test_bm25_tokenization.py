from src.retrieval.bm25 import _tokenize


def test_tokenize_russian_normalization_yo_e():
    tokens = _tokenize("Ёжик в тумане и ель")
    assert "ежик" in tokens


def test_tokenize_filters_stopwords_and_short_tokens():
    tokens = _tokenize("и в на это auth проблема")
    assert "и" not in tokens
    assert "в" not in tokens
    assert "проблема" in tokens
