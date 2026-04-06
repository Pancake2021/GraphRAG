import sys
import types

from src.eval.ragas_eval import evaluate_with_ragas


class _FakeDataset:
    @staticmethod
    def from_list(rows):
        return rows


class _FakeResult:
    def __init__(self, rows):
        self.scores = rows


def test_evaluate_with_ragas_aggregates_scores(monkeypatch):
    fake_ragas = types.ModuleType("ragas")
    fake_metrics = types.ModuleType("ragas.metrics")
    fake_dataset_module = types.ModuleType("datasets")
    fake_lc_ollama = types.ModuleType("langchain_ollama")

    fake_metrics.faithfulness = object()
    fake_metrics.context_precision = object()
    fake_metrics.context_recall = object()

    def _fake_evaluate(*, dataset, metrics, llm, raise_exceptions, show_progress):
        assert dataset
        assert len(metrics) == 3
        _ = llm
        _ = raise_exceptions
        _ = show_progress
        return _FakeResult(
            [
                {"faithfulness": 0.8, "context_precision": 0.6, "context_recall": 0.7},
                {"faithfulness": 0.6, "context_precision": 0.4, "context_recall": 0.5},
            ]
        )

    class _FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_ragas.evaluate = _fake_evaluate
    fake_dataset_module.Dataset = _FakeDataset
    fake_lc_ollama.ChatOllama = _FakeChatOllama

    monkeypatch.setitem(sys.modules, "ragas", fake_ragas)
    monkeypatch.setitem(sys.modules, "ragas.metrics", fake_metrics)
    monkeypatch.setitem(sys.modules, "datasets", fake_dataset_module)
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_lc_ollama)

    payload = evaluate_with_ragas(
        [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"], "reference": "ref"}],
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
        show_progress=False,
    )

    assert payload.summary["faithfulness"] == 0.7
    assert payload.summary["context_precision"] == 0.5
    assert payload.summary["context_recall"] == 0.6
    assert len(payload.rows) == 2

