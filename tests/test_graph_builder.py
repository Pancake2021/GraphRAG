from src.graph.builder import build_graph
from src.models import ChunkRecord, DocumentRecord, Relation


def test_graph_builder_adds_nodes_and_edges():
    docs = [DocumentRecord(id="d1", doc_type="chat", text="x", metadata={})]
    chunks = [
        ChunkRecord(
            id="d1_c0",
            doc_id="d1",
            text="Ivan solved auth bug",
            entities={"Person": ["Ivan Petrov"]},
            relations=[
                Relation(
                    subject="Ivan Petrov",
                    predicate="SOLVED",
                    object="Auth bug",
                    confidence=0.9,
                    source_chunk_id="d1_c0",
                )
            ],
            metadata={},
        )
    ]

    g = build_graph(docs, chunks)
    assert "d1" in g
    assert "Ivan Petrov" in g
    assert "Auth bug" in g
    assert g.number_of_edges() >= 2


def test_graph_builder_idempotent_relations():
    docs = [DocumentRecord(id="d1", doc_type="chat", text="x", metadata={})]
    rel = Relation(
        subject="Ivan Petrov",
        predicate="SOLVED",
        object="Auth bug",
        confidence=0.9,
        source_chunk_id="d1_c0",
    )
    chunks = [
        ChunkRecord(id="d1_c0", doc_id="d1", text="x", entities={}, relations=[rel, rel], metadata={})
    ]

    g = build_graph(docs, chunks)
    solved_edges = [e for e in g.edges(data=True) if e[2].get("type") == "SOLVED"]
    assert len(solved_edges) == 1
