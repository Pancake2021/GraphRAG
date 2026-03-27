from pathlib import Path

from src.governance.pii import pseudonymize_text
from src.io_utils import read_jsonl


def test_pseudonymize_is_stable_and_does_not_store_raw_value(tmp_path: Path):
    mapping_path = tmp_path / "pii_mapping.jsonl"
    text = "Свяжитесь с Иван Иванов по почте ivan@example.com и телефону +7 999 123 45 67."

    redacted_first, created_first = pseudonymize_text(text, mapping_path)
    redacted_second, created_second = pseudonymize_text(text, mapping_path)

    assert "[EMAIL_" in redacted_first
    assert "[PHONE_" in redacted_first
    assert redacted_first == redacted_second
    assert created_first
    assert created_second == []

    rows = read_jsonl(mapping_path)
    assert rows
    for row in rows:
        assert "raw_hash" in row
        assert "raw_value" not in row
        assert row.get("pseudonym", "").startswith("[")
    assert "ivan@example.com" not in mapping_path.read_text(encoding="utf-8")

