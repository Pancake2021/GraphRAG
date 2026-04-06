from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RagasEvaluationPayload:
    summary: dict[str, float]
    rows: list[dict[str, Any]]


def evaluate_with_ragas(
    rows: list[dict[str, Any]],
    *,
    ollama_base_url: str,
    ollama_model: str,
    show_progress: bool = True,
) -> RagasEvaluationPayload:
    if not rows:
        raise ValueError("RAGAS evaluation requires non-empty rows")

    from datasets import Dataset
    from langchain_ollama import ChatOllama
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_base_url,
        temperature=0.0,
    )
    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
        raise_exceptions=False,
        show_progress=show_progress,
    )

    score_rows: list[dict[str, Any]] = []
    if hasattr(result, "scores"):
        score_rows = list(result.scores)
    elif hasattr(result, "to_pandas"):
        score_rows = result.to_pandas().to_dict("records")
    else:
        raise RuntimeError("Unexpected RAGAS result format")

    metric_keys = ("faithfulness", "context_precision", "context_recall")
    summary: dict[str, float] = {}
    for key in metric_keys:
        vals: list[float] = []
        for row in score_rows:
            value = row.get(key)
            if isinstance(value, (float, int)):
                vals.append(float(value))
        summary[key] = (sum(vals) / len(vals)) if vals else 0.0

    return RagasEvaluationPayload(summary=summary, rows=score_rows)

