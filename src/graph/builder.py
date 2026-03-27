from __future__ import annotations

import hashlib

import networkx as nx

from src.models import ChunkRecord, DocumentRecord, Relation


def _guess_node_type(value: str) -> str:
    low = value.lower()
    if "meeting" in low or "sprint" in low:
        return "Meeting"
    if any(x in low for x in ["bug", "issue", "problem", "deadline", "task"]):
        return "Problem"
    if any(x in low for x in ["fix", "solution", "upgrade", "approach", "idea"]):
        return "Solution"
    if any(c.isdigit() for c in value) and any(ch in value for ch in "-/."):
        return "Date"
    words = value.split()
    if len(words) >= 2 and all(w[:1].isupper() for w in words if w):
        return "Person"
    return "Decision"


def _add_edge_idempotent(g: nx.MultiDiGraph, source: str, target: str, **attrs) -> None:
    edge_signature = f"{source}|{target}|{attrs.get('type')}|{attrs.get('source_chunk_id', '')}"
    edge_id = hashlib.md5(edge_signature.encode("utf-8")).hexdigest()

    for _, _, data in g.edges(source, data=True):
        if data.get("edge_id") == edge_id:
            return
    g.add_edge(source, target, edge_id=edge_id, **attrs)


def build_graph(docs: list[DocumentRecord], chunks: list[ChunkRecord]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for doc in docs:
        g.add_node(doc.id, type="Document", label=doc.id, metadata=doc.metadata)

    for chunk in chunks:
        for ent_type, values in chunk.entities.items():
            for value in values:
                node_type = ent_type if ent_type in {"Person", "Organization", "Date"} else _guess_node_type(value)
                g.add_node(value, type=node_type, label=value)
                _add_edge_idempotent(
                    g,
                    value,
                    chunk.doc_id,
                    type="MENTIONED_IN",
                    confidence=1.0,
                    source_chunk_id=chunk.id,
                )

        for rel in chunk.relations:
            _add_relation(g, rel)

    return g


def _add_relation(g: nx.MultiDiGraph, rel: Relation) -> None:
    if rel.subject not in g:
        g.add_node(rel.subject, type=_guess_node_type(rel.subject), label=rel.subject)
    if rel.object not in g:
        g.add_node(rel.object, type=_guess_node_type(rel.object), label=rel.object)

    _add_edge_idempotent(
        g,
        rel.subject,
        rel.object,
        type=rel.predicate,
        confidence=rel.confidence,
        source_chunk_id=rel.source_chunk_id,
    )
