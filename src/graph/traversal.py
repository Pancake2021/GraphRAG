from __future__ import annotations

import networkx as nx


def get_neighbors(graph: nx.MultiDiGraph, nodes: list[str], depth: int = 1) -> set[str]:
    visited: set[str] = set(nodes)
    frontier: set[str] = set(nodes)
    for _ in range(max(1, depth)):
        next_frontier: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            next_frontier.update(graph.successors(node))
            next_frontier.update(graph.predecessors(node))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def extract_subgraph(graph: nx.MultiDiGraph, nodes: list[str], depth: int = 1) -> nx.MultiDiGraph:
    neighborhood = get_neighbors(graph, nodes, depth=depth)
    return graph.subgraph(neighborhood).copy()
