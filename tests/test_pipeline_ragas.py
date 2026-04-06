import json
from pathlib import Path

from src import pipeline
from src.models import AnswerResult, ChunkRecord, EvalQuestion


class _FakeLLM:
    def ensure_model_ready(self):
        return None


def test_cmd_evaluate_ragas_writes_report(monkeypatch, tmp_path: Path):
    s = pipeline.settings
    s.processed_data_dir = tmp_path / "processed"
    s.eval_data_dir = tmp_path / "eval"
    s.ragas_report_path = tmp_path / "processed" / "ragas_evaluation.json"
    s.ensure_dirs()

    monkeypatch.setattr(pipeline, "_llm", lambda: _FakeLLM())
    monkeypatch.setattr(
        pipeline,
        "_load_cases",
        lambda _path: [EvalQuestion(id="q1", question="Кто решил баг?", ground_truth="Иван решил баг.")],
    )

    chunk = ChunkRecord(
        id="c1",
        doc_id="d1",
        text="Иван решил баг в auth.",
        entities={"Person": ["Иван"]},
        relations=[],
        metadata={},
    )
    monkeypatch.setattr(pipeline, "_retrieve", lambda _q, top_k, use_graph=True: ([chunk], []))
    monkeypatch.setattr(
        pipeline,
        "_safe_answer",
        lambda q, chunks, graph_context, lang, metrics=None: AnswerResult(
            answer="Иван решил баг.",
            sources=[{"chunk_id": chunks[0].id, "doc_id": chunks[0].doc_id}],
            graph_context=graph_context,
            retrieved_chunks=chunks,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_with_ragas",
        lambda rows, **_kwargs: type(
            "_P",
            (),
            {
                "summary": {"faithfulness": 0.91, "context_precision": 0.85, "context_recall": 0.88},
                "rows": [{"faithfulness": 0.91, "context_precision": 0.85, "context_recall": 0.88}],
            },
        )(),
    )

    report = pipeline.cmd_evaluate_ragas(s.eval_data_dir / "questions.json", top_k=3, lang="ru")
    assert report["summary"]["faithfulness"] == 0.91
    assert s.ragas_report_path.exists()

    payload = json.loads(s.ragas_report_path.read_text(encoding="utf-8"))
    assert payload["num_questions"] == 1

