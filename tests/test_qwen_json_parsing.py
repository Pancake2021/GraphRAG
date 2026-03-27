from src.llm.qwen_client import QwenClient


def test_json_candidate_parser_handles_markdown_and_trailing_text():
    client = QwenClient(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")
    text = """Ответ:\n```json\n{\"documents\":[{\"doc_type\":\"chat\",\"text\":\"ok\",\"metadata\":{}}]}\n```\nГотово."""
    parsed = client._parse_json_candidates(text)  # noqa: SLF001
    assert parsed is not None
    assert "documents" in parsed

