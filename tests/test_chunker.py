import pytest

from src.preprocessing.chunker import chunk_text, normalize_text


def test_normalize_text_compacts_spaces():
    assert normalize_text(" A   B\n\nC ") == "A B C"


def test_chunk_text_overlap_boundaries():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_empty():
    assert chunk_text("   ") == []


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=5, overlap=5)
