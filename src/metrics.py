from __future__ import annotations

from src.models import AnswerResult, EvalQuestion


def precision_at_k(retrieved_texts: list[str], ground_truth: str, k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    gt_terms = set(ground_truth.lower().split())
    relevant = 0
    for txt in top:
        terms = set(txt.lower().split())
        if gt_terms & terms:
            relevant += 1
    return relevant / len(top)


def graph_coverage(graph_added_chunks: int, total_questions: int) -> float:
    if total_questions == 0:
        return 0.0
    return graph_added_chunks / total_questions


def simple_answer_relevance(answer: str, ground_truth: str) -> float:
    a = set(answer.lower().split())
    g = set(ground_truth.lower().split())
    if not a or not g:
        return 0.0
    return len(a & g) / len(g)


def evaluate_question(
    question: EvalQuestion,
    baseline_result: AnswerResult,
    graphrag_result: AnswerResult,
    top_k: int,
) -> dict:
    baseline_texts = [c.text for c in baseline_result.retrieved_chunks]
    graph_texts = [c.text for c in graphrag_result.retrieved_chunks]

    return {
        "id": question.id,
        "question": question.question,
        "precision_at_k_baseline": precision_at_k(baseline_texts, question.ground_truth, top_k),
        "precision_at_k_graphrag": precision_at_k(graph_texts, question.ground_truth, top_k),
        "answer_relevance_baseline": simple_answer_relevance(baseline_result.answer, question.ground_truth),
        "answer_relevance_graphrag": simple_answer_relevance(graphrag_result.answer, question.ground_truth),
        "graph_added_context": max(0, len(graph_texts) - len(baseline_texts)),
    }
