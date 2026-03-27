from pathlib import Path

from src import pipeline
from src.models import ChunkRecord, DocumentRecord, Relation


class _FakeLLM:
    def ensure_model_ready(self):
        return None

    def generate_json(self, prompt, system=None, temperature=0.0):
        if "Generate synthetic" in prompt:
            return {
                "documents": [
                    {
                        "doc_type": "chat",
                        "text": "Ivan Petrov and Maria Sidorova discussed auth bug and solution.",
                        "metadata": {"date": "2025-01-15"},
                    }
                ]
            }
        return {
            "relations": [
                {
                    "subject": "Ivan Petrov",
                    "predicate": "SOLVED",
                    "object": "Auth bug",
                    "confidence": 0.9,
                }
            ]
        }

    def generate(self, prompt, system=None, temperature=0.1):
        return "Ivan Petrov solved Auth bug. Sources: d1_c0"


class _FakeStore:
    def __init__(self, *args, **kwargs):
        self.rows = []

    def upsert_chunks(self, chunks):
        self.rows = chunks

    def search(self, query, top_k=10):
        return [("doc_test_c0", 0.9, {"doc_id": "doc_test"})]


def test_end_to_end_query_flow(monkeypatch, tmp_path: Path):
    s = pipeline.settings
    s.raw_data_dir = tmp_path / "raw"
    s.processed_data_dir = tmp_path / "processed"
    s.eval_data_dir = tmp_path / "eval"
    s.chroma_dir = tmp_path / "chroma"
    s.graph_pickle_path = tmp_path / "processed" / "graph.gpickle"
    s.graph_json_path = tmp_path / "processed" / "graph.json"
    s.ensure_dirs()

    monkeypatch.setattr(pipeline, "_llm", lambda: _FakeLLM())

    docs = [DocumentRecord(id="doc_test", doc_type="chat", text="Ivan solved auth bug.", metadata={})]
    chunks = [
        ChunkRecord(
            id="doc_test_c0",
            doc_id="doc_test",
            text="Ivan solved auth bug.",
            entities={"Person": ["Ivan Petrov"]},
            relations=[],
            metadata={"doc_type": "chat"},
        )
    ]

    monkeypatch.setattr(pipeline, "_load_docs", lambda: docs)
    monkeypatch.setattr(pipeline, "documents_to_chunks", lambda *_args, **_kwargs: chunks)
    monkeypatch.setattr(pipeline, "extract_entities", lambda _text: {"Person": ["Ivan Petrov"], "Organization": [], "Date": []})
    monkeypatch.setattr(
        pipeline,
        "extract_relations",
        lambda _llm, chunk_id, _text, lang="ru": [
            Relation(
                subject="Ivan Petrov",
                predicate="SOLVED",
                object="Auth bug",
                confidence=0.9,
                source_chunk_id=chunk_id,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "VectorStore", _FakeStore)

    pipeline.cmd_build_index()

    monkeypatch.setattr(pipeline, "_load_chunks", lambda: chunks)
    monkeypatch.setattr(pipeline, "_answer", lambda q, c, g: pipeline.AnswerResult(answer="ok", sources=[{"chunk_id": c[0].id, "doc_id": c[0].doc_id}], graph_context=g, retrieved_chunks=c))

    retrieved, graph_context = pipeline._retrieve("Who solved bug?", top_k=5, use_graph=True)
    assert retrieved
    assert isinstance(graph_context, list)
