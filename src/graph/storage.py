from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx


def save_graph(graph: nx.MultiDiGraph, pickle_path: Path, json_path: Path) -> None:
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(graph, f)

    graph_json = nx.node_link_data(graph)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)


def load_graph(pickle_path: Path) -> nx.MultiDiGraph:
    if not pickle_path.exists():
        return nx.MultiDiGraph()
    with pickle_path.open("rb") as f:
        return pickle.load(f)
