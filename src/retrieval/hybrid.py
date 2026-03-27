from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.models import ChunkRecord


def reciprocal_rank_fusion(
    bm25_results: list[tuple[ChunkRecord, float]],
    vector_results: list[tuple[ChunkRecord, float]],
    k: int = 60,
) -> list[ChunkRecord]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, ChunkRecord] = {}

    for rank, (chunk, _) in enumerate(bm25_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        by_id[chunk.id] = chunk
    for rank, (chunk, _) in enumerate(vector_results, start=1):
        scores[chunk.id] += 1.0 / (k + rank)
        by_id[chunk.id] = chunk

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [by_id[cid] for cid, _ in ordered]


def deduplicate_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    seen: set[str] = set()
    out: list[ChunkRecord] = []
    for c in chunks:
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


def graph_expand_chunks(
    graph: nx.MultiDiGraph,
    fused_chunks: list[ChunkRecord],
    all_chunks_by_id: dict[str, ChunkRecord],
    depth: int = 1,
) -> list[ChunkRecord]:
    expanded: list[ChunkRecord] = []
    for chunk in fused_chunks:
        entities = [v for vals in chunk.entities.values() for v in vals]
        neighborhood = set(entities)
        frontier = set(entities)

        for _ in range(max(1, depth)):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in graph:
                    continue
                next_frontier.update(graph.successors(node))
                next_frontier.update(graph.predecessors(node))
            next_frontier -= neighborhood
            neighborhood.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        for node in neighborhood:
            if node not in graph:
                continue
            for _, _, data in graph.in_edges(node, data=True):
                chunk_id = data.get("source_chunk_id")
                if chunk_id and chunk_id in all_chunks_by_id:
                    expanded.append(all_chunks_by_id[chunk_id])
            for _, _, data in graph.out_edges(node, data=True):
                chunk_id = data.get("source_chunk_id")
                if chunk_id and chunk_id in all_chunks_by_id:
                    expanded.append(all_chunks_by_id[chunk_id])

    return deduplicate_chunks(expanded)
