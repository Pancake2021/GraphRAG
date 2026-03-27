from src.i18n import detect_language, normalize_predicate, resolve_language, synthetic_prompt


def test_detect_language_ru_en():
    assert detect_language("Привет как дела") == "ru"
    assert detect_language("hello world") == "en"


def test_resolve_language_preferred_and_fallback():
    assert resolve_language("ru", "hello", "en", ["ru", "en"]) == "ru"
    assert resolve_language(None, "hello", "ru", ["ru", "en"]) == "en"
    assert resolve_language(None, "привет", "en", ["ru", "en"]) == "ru"


def test_normalize_predicate_russian_to_en():
    assert normalize_predicate("РЕШИЛ") == "SOLVED"
    assert normalize_predicate("ПРЕДЛОЖИЛА") == "PROPOSED"
    assert normalize_predicate("RELATED_TO") == "RELATED_TO"


def test_synthetic_prompt_russian_default():
    prompt = synthetic_prompt("ru", count=3)
    assert "Сгенерируй" in prompt
    assert "Количество: 3" in prompt
