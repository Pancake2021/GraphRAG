import networkx as nx

from src.models import ChunkRecord
from src.retrieval.hybrid import deduplicate_chunks, graph_expand_chunks, reciprocal_rank_fusion


def _chunk(cid: str, entities: dict[str, list[str]] | None = None):
    return ChunkRecord(id=cid, doc_id="d1", text=f"text {cid}", entities=entities or {}, relations=[], metadata={})


def test_rrf_dedup_ordered():
    a = _chunk("a")
    b = _chunk("b")
    c = _chunk("c")
    out = reciprocal_rank_fusion([(a, 1.0), (b, 0.5)], [(b, 1.0), (c, 0.4)])
    ids = [x.id for x in out]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_deduplicate_chunks():
    a = _chunk("a")
    b = _chunk("b")
    out = deduplicate_chunks([a, b, a])
    assert [x.id for x in out] == ["a", "b"]


def test_graph_expand_chunks():
    g = nx.MultiDiGraph()
    g.add_edge("Ivan Petrov", "Auth bug", source_chunk_id="c2", type="SOLVED")

    c1 = _chunk("c1", {"Person": ["Ivan Petrov"]})
    c2 = _chunk("c2")

    expanded = graph_expand_chunks(g, [c1], {"c1": c1, "c2": c2}, depth=1)
    assert any(c.id == "c2" for c in expanded)
