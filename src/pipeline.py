from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import settings
from src.data_gen.synthetic import generate_synthetic_documents
from src.extraction.ner import extract_entities
from src.extraction.normalization import normalize_entities
from src.extraction.relation_quality import validate_relations
from src.extraction.relations import extract_relations
from src.governance.pii import pseudonymize_text
from src.governance.retention import apply_retention, build_retention_candidates, retention_report
from src.graph.builder import build_graph
from src.graph.storage import load_graph, save_graph
from src.i18n import Lang, node_type_label, resolve_language
from src.i18n import answer_prompt as i18n_answer_prompt
from src.ingest.state import apply_versioning, latest_doc_ids, load_manifest, save_manifest
from src.io_utils import read_json, read_jsonl, write_json, write_jsonl
from src.llm.qwen_client import QwenClient
from src.logging_config import setup_logging
from src.metrics import evaluate_question, graph_coverage
from src.models import AnswerResult, ChunkRecord, DocumentRecord, EvalQuestion
from src.ops.health import health_live, health_ready
from src.ops.metrics_store import MetricsStore, evaluate_alerts
from src.preprocessing.chunker import documents_to_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import deduplicate_chunks, graph_expand_chunks, reciprocal_rank_fusion
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _active_lang(preferred: str | None, text_hint: str | None = None) -> Lang:
    return resolve_language(
        preferred=preferred,
        text_hint=text_hint,
        default_language=settings.default_language,
        supported_languages=settings.supported_languages,
    )


def _llm() -> QwenClient:
    return QwenClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
        retries=settings.ollama_retries,
        retry_backoff_sec=settings.ollama_retry_backoff_sec,
    )


def _metrics() -> MetricsStore:
    return MetricsStore(settings.ops_metrics_path, settings.ops_metrics_prom_path)


def _profile(name: str | None = None) -> dict:
    profile_name = name or settings.app_env
    path = settings.profiles_dir / f"{profile_name}.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {
            "retention_days": {"raw": 30, "processed": 60, "logs": 14, "eval": 90},
            "alerts": {
                "max_stage_latency_sec": 30,
                "min_relation_valid_rate": 0.7,
                "max_error_rate": 0.2,
            },
        }
    return payload


def _load_raw_docs() -> list[DocumentRecord]:
    rows = read_jsonl(settings.raw_data_dir / "documents.jsonl")
    return [DocumentRecord.model_validate(r) for r in rows]


def _load_docs() -> list[DocumentRecord]:
    # Backward-compatible alias used in older tests/monkeypatches.
    return _load_raw_docs()


def _load_latest_docs() -> list[DocumentRecord]:
    rows = read_jsonl(settings.documents_latest_path)
    return [DocumentRecord.model_validate(r) for r in rows]


def _load_chunks() -> list[ChunkRecord]:
    rows = read_jsonl(settings.processed_data_dir / "chunks.jsonl")
    return [ChunkRecord.model_validate(r) for r in rows]


def _persist_managed_docs(new_docs: list[DocumentRecord]) -> list[DocumentRecord]:
    existing = {
        doc.id: doc
        for doc in [DocumentRecord.model_validate(r) for r in read_jsonl(settings.managed_documents_path)]
    }
    for doc in new_docs:
        existing[doc.id] = doc
    merged = list(existing.values())
    write_jsonl(settings.managed_documents_path, merged)
    return merged


def _apply_pii(docs: list[DocumentRecord]) -> tuple[list[DocumentRecord], int]:
    redacted_count = 0
    out: list[DocumentRecord] = []
    for doc in docs:
        raw_text = doc.text
        redacted, created = pseudonymize_text(raw_text, settings.pii_mapping_path)
        doc.source_text_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        doc.text = redacted
        doc.metadata = {
            **doc.metadata,
            "pii_redacted": len(created),
            "pii_safe": True,
        }
        redacted_count += len(created)
        out.append(doc)
    return out, redacted_count


def _ingest_documents(raw_docs: list[DocumentRecord], incremental: bool) -> tuple[list[DocumentRecord], list[DocumentRecord], dict]:
    docs_pii, pii_count = _apply_pii(raw_docs)

    manifest = load_manifest(settings.ingest_manifest_path)
    processed, manifest, summary = apply_versioning(docs_pii, manifest)
    save_manifest(settings.ingest_manifest_path, manifest)

    _persist_managed_docs(processed)

    latest_ids = latest_doc_ids(manifest)
    all_managed = [DocumentRecord.model_validate(r) for r in read_jsonl(settings.managed_documents_path)]
    latest_docs = [d for d in all_managed if d.id in latest_ids and d.ingest_status != "invalid"]

    if incremental:
        changed = [d for d in processed if d.ingest_status in {"new", "updated"}]
    else:
        changed = latest_docs

    summary["pii_redacted"] = pii_count
    summary["latest_docs"] = len(latest_docs)
    summary["changed_docs"] = len(changed)
    return latest_docs, changed, summary


def _is_llm_timeout_error(exc: Exception) -> bool:
    return "timed out" in str(exc).lower()


def _index_docs(
    latest_docs: list[DocumentRecord],
    changed_docs: list[DocumentRecord],
    incremental: bool,
    metrics: MetricsStore | None = None,
) -> dict:
    existing_chunks = _load_chunks() if incremental else []
    latest_doc_ids_set = {d.id for d in latest_docs}
    changed_ids = {d.id for d in changed_docs}

    base_chunks: list[ChunkRecord] = []
    if incremental:
        base_chunks = [
            c
            for c in existing_chunks
            if c.doc_id in latest_doc_ids_set and c.doc_id not in changed_ids
        ]

    new_chunks = documents_to_chunks(changed_docs, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

    total_invalid_relations = 0
    total_valid_relations = 0
    total_entity_merges = 0

    llm_client = _llm()
    for chunk in new_chunks:
        chunk_lang = _active_lang(None, chunk.text)
        raw_entities = extract_entities(chunk.text)
        norm_entities, alias_map, merges = normalize_entities(raw_entities)
        chunk.entities = norm_entities
        chunk.metadata = {
            **chunk.metadata,
            "entity_alias_map": alias_map,
            "entity_merges": merges,
        }
        total_entity_merges += merges

        try:
            extracted = extract_relations(llm_client, chunk.id, chunk.text, lang=chunk_lang)
        except RuntimeError as exc:
            if _is_llm_timeout_error(exc):
                logger.warning("Relation extraction timed out on chunk=%s; continuing with empty relations", chunk.id)
                if metrics:
                    metrics.inc("llm_timeout_count", 1)
                extracted = []
            else:
                raise
        validation = validate_relations(extracted, min_confidence=0.5)
        total_valid_relations += len(validation.valid)
        total_invalid_relations += len(validation.invalid)
        if metrics and validation.invalid:
            metrics.inc("schema_validation_fail_count", len(validation.invalid))
        chunk.relations = validation.valid
        chunk.metadata = {
            **chunk.metadata,
            "invalid_relations": validation.invalid,
        }

    combined = deduplicate_chunks(base_chunks + new_chunks)

    write_jsonl(settings.processed_data_dir / "chunks.jsonl", combined)
    write_jsonl(settings.documents_latest_path, latest_docs)

    graph = build_graph(latest_docs, combined)
    save_graph(graph, settings.graph_pickle_path, settings.graph_json_path)

    store = VectorStore(str(settings.chroma_dir), settings.chroma_collection)

    if incremental:
        stale_chunk_ids = [
            c.id
            for c in existing_chunks
            if c.doc_id not in latest_doc_ids_set or c.doc_id in changed_ids
        ]
        if hasattr(store, "delete_chunks"):
            store.delete_chunks(stale_chunk_ids)
        store.upsert_chunks(new_chunks)
    else:
        if hasattr(store, "reset_collection"):
            store.reset_collection()
        store.upsert_chunks(combined)

    rel_total = total_valid_relations + total_invalid_relations
    relation_valid_rate = (total_valid_relations / rel_total) if rel_total else 1.0

    return {
        "latest_docs": len(latest_docs),
        "indexed_chunks": len(combined),
        "new_chunks": len(new_chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "relation_valid_rate": relation_valid_rate,
        "invalid_relations": total_invalid_relations,
        "valid_relations": total_valid_relations,
        "entity_norm_merge_rate": (total_entity_merges / max(1, len(new_chunks))),
    }


def _retrieve(query: str, top_k: int, use_graph: bool = True) -> tuple[list[ChunkRecord], list[dict]]:
    logger.info("Retrieve started: query=%s top_k=%s use_graph=%s", query, top_k, use_graph)
    chunks = _load_chunks()
    if not chunks:
        raise RuntimeError("No processed chunks found. Run ingest-full or build-index first.")

    by_id = {c.id: c for c in chunks}
    bm25 = BM25Retriever(chunks)
    store = VectorStore(str(settings.chroma_dir), settings.chroma_collection)

    bm25_rows = bm25.search(query, top_k=10)
    vector_hits = store.search(query, top_k=10)

    vector_rows = []
    for cid, score, _meta in vector_hits:
        chunk = by_id.get(cid)
        if chunk:
            vector_rows.append((chunk, score))

    fused = reciprocal_rank_fusion(bm25_rows, vector_rows)[:top_k]
    graph_context: list[dict] = []

    if not use_graph:
        return fused, graph_context

    graph = load_graph(settings.graph_pickle_path)
    expanded = graph_expand_chunks(graph, fused, by_id, depth=1)

    for c in fused:
        graph_context.append({"chunk_id": c.id, "entities": c.entities})

    return deduplicate_chunks(fused + expanded)[: max(top_k, len(fused))], graph_context


def _answer(query: str, chunks: list[ChunkRecord], graph_context: list[dict], lang: Lang = "ru") -> AnswerResult:
    client = _llm()
    context = "\n\n".join([f"[{c.id}] {c.text}" for c in chunks])
    graph_ctx = json.dumps(graph_context, ensure_ascii=False)
    prompt = i18n_answer_prompt(lang=lang, query=query, graph_ctx_json=graph_ctx, chunks_text=context)
    answer = client.generate(prompt, temperature=0.1)
    sources = [{"chunk_id": c.id, "doc_id": c.doc_id} for c in chunks]
    return AnswerResult(
        answer=answer,
        sources=sources,
        graph_context=graph_context,
        retrieved_chunks=chunks,
    )


def _fallback_answer(query: str, chunks: list[ChunkRecord], graph_context: list[dict], lang: Lang) -> AnswerResult:
    if lang == "ru":
        answer = (
            "LLM недоступна по таймауту. Возвращаю контекст без генерации: "
            + (chunks[0].text[:320] if chunks else "контекст не найден.")
        )
    else:
        answer = (
            "LLM timed out. Returning retrieval context without generation: "
            + (chunks[0].text[:320] if chunks else "no context found.")
        )
    sources = [{"chunk_id": c.id, "doc_id": c.doc_id} for c in chunks]
    return AnswerResult(
        answer=answer,
        sources=sources,
        graph_context=graph_context,
        retrieved_chunks=chunks,
    )


def _safe_answer(
    query: str,
    chunks: list[ChunkRecord],
    graph_context: list[dict],
    lang: Lang,
    metrics: MetricsStore | None = None,
) -> AnswerResult:
    try:
        try:
            return _answer(query, chunks, graph_context, lang)
        except TypeError:
            # Backward-compatible path for monkeypatched _answer(query, chunks, graph_context)
            return _answer(query, chunks, graph_context)  # type: ignore[misc,call-arg]
    except RuntimeError as exc:
        if _is_llm_timeout_error(exc):
            logger.warning("Answer generation timed out for query=%s", query)
            if metrics:
                metrics.inc("llm_timeout_count", 1)
            return _fallback_answer(query, chunks, graph_context, lang)
        raise


def _explain_graph_context(graph_context: list[dict], lang: Lang) -> list[str]:
    lines: list[str] = []
    for item in graph_context:
        chunk_id = item.get("chunk_id", "")
        entities = item.get("entities", {})
        if not isinstance(entities, dict):
            continue
        parts: list[str] = []
        for node_type, values in entities.items():
            if not values:
                continue
            label = node_type_label(node_type, lang)
            parts.append(f"{label}: {', '.join(values)}")
        if parts:
            lines.append(f"{chunk_id}: " + " | ".join(parts))
    return lines


def _load_cases(path: Path) -> list[EvalQuestion]:
    rows = read_jsonl(path)
    return [EvalQuestion.model_validate(r) for r in rows]


def _write_interview_report(quality: dict | None, benchmark: dict | None) -> dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_platform": platform.platform(),
        "data_governance": {
            "pii_mode": "pseudonymization_with_audit_mapping",
            "manifest": str(settings.ingest_manifest_path),
            "retention_report": str(settings.retention_report_path),
        },
        "quality": quality or {},
        "benchmark": benchmark or {},
        "notes": [
            "EN-compatible internal graph schema (predicates remain EN codes)",
            "RU-first prompts and evaluation",
            "Incremental ingest with doc versioning and idempotency",
        ],
    }
    write_json(settings.interview_report_path, payload)
    return payload


def cmd_generate_data(n: int, lang: str | None = None) -> None:
    active_lang = _active_lang(lang)
    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("generate_data"):
        settings.ensure_dirs()
        client = _llm()
        client.ensure_model_ready()

        docs = generate_synthetic_documents(
            client,
            count=n,
            batch_size=settings.synthetic_batch_size,
            lang=active_lang,
        )
        write_jsonl(settings.raw_data_dir / "documents.jsonl", docs)
    m.flush()
    if active_lang == "ru":
        print(f"Сгенерировано {len(docs)} документов: {settings.raw_data_dir / 'documents.jsonl'}")
    else:
        print(f"Generated {len(docs)} documents at {settings.raw_data_dir / 'documents.jsonl'}")


def cmd_ingest_full(n: int, lang: str | None = None) -> None:
    cmd_generate_data(n=n, lang=lang)
    raw_docs = _load_raw_docs()
    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("ingest_full"):
        latest_docs, changed_docs, ingest_summary = _ingest_documents(raw_docs, incremental=False)
        index_summary = _index_docs(latest_docs, changed_docs, incremental=False, metrics=m)

    m.set_quality("relation_valid_rate", index_summary["relation_valid_rate"])
    m.set_quality("entity_norm_merge_rate", index_summary["entity_norm_merge_rate"])
    m.inc("relation_extracted_count", index_summary["valid_relations"])
    m.flush()

    print(json.dumps({"ingest": ingest_summary, "index": index_summary}, ensure_ascii=False, indent=2))


def cmd_ingest_incremental(lang: str | None = None) -> None:
    _ = _active_lang(lang)
    raw_docs = _load_raw_docs()
    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("ingest_incremental"):
        latest_docs, changed_docs, ingest_summary = _ingest_documents(raw_docs, incremental=True)
        index_summary = _index_docs(latest_docs, changed_docs, incremental=True, metrics=m)

    m.set_quality("relation_valid_rate", index_summary["relation_valid_rate"])
    m.set_quality("entity_norm_merge_rate", index_summary["entity_norm_merge_rate"])
    m.inc("relation_extracted_count", index_summary["valid_relations"])
    m.flush()

    print(json.dumps({"ingest": ingest_summary, "index": index_summary}, ensure_ascii=False, indent=2))


def cmd_build_index() -> None:
    raw_docs = _load_docs()
    latest_docs, changed_docs, _ = _ingest_documents(raw_docs, incremental=False)
    m = _metrics()
    m.inc("operation_count", 1)
    summary = _index_docs(latest_docs, changed_docs, incremental=False, metrics=m)
    m.set_quality("relation_valid_rate", summary["relation_valid_rate"])
    m.set_quality("entity_norm_merge_rate", summary["entity_norm_merge_rate"])
    m.inc("relation_extracted_count", summary["valid_relations"])
    m.flush()
    print(f"Индексация завершена: chunks={summary['indexed_chunks']} nodes={summary['graph_nodes']} edges={summary['graph_edges']}")


def cmd_query(query: str, top_k: int, lang: str | None = None) -> None:
    active_lang = _active_lang(lang, text_hint=query)
    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("query"):
        _llm().ensure_model_ready()
        chunks, graph_context = _retrieve(query, top_k=top_k, use_graph=True)
        result = _safe_answer(query, chunks, graph_context, lang=active_lang, metrics=m)

    m.flush()

    if active_lang == "ru":
        print("\n=== Ответ ===\n")
        print(result.answer)
        print("\n=== Источники ===")
    else:
        print("\n=== Answer ===\n")
        print(result.answer)
        print("\n=== Sources ===")

    for s in result.sources:
        print(f"- chunk={s['chunk_id']} doc={s['doc_id']}")

    explain = _explain_graph_context(graph_context, active_lang)
    if explain:
        if active_lang == "ru":
            print("\n=== Контекст графа ===")
        else:
            print("\n=== Graph context ===")
        for line in explain:
            print(f"- {line}")


def cmd_evaluate(questions_path: Path, top_k: int, lang: str | None = None) -> dict:
    _llm().ensure_model_ready()
    questions = _load_cases(questions_path)
    if not questions:
        raise RuntimeError(f"No questions found in {questions_path}")

    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("evaluate"):
        rows = []
        graph_added_total = 0
        for q in questions:
            q_lang = _active_lang(lang, text_hint=q.question)
            base_chunks, _ = _retrieve(q.question, top_k=top_k, use_graph=False)
            graph_chunks, graph_ctx = _retrieve(q.question, top_k=top_k, use_graph=True)

            base_ans = _safe_answer(q.question, base_chunks, graph_context=[], lang=q_lang, metrics=m)
            graph_ans = _safe_answer(
                q.question,
                graph_chunks,
                graph_context=graph_ctx,
                lang=q_lang,
                metrics=m,
            )

            row = evaluate_question(q, base_ans, graph_ans, top_k=top_k)
            graph_added_total += int(row["graph_added_context"] > 0)
            rows.append(row)

        cov = graph_coverage(graph_added_total, len(questions))
        summary = {
            "num_questions": len(questions),
            "graph_coverage": cov,
            "avg_precision_at_k_baseline": sum(r["precision_at_k_baseline"] for r in rows) / len(rows),
            "avg_precision_at_k_graphrag": sum(r["precision_at_k_graphrag"] for r in rows) / len(rows),
            "avg_answer_relevance_baseline": sum(r["answer_relevance_baseline"] for r in rows) / len(rows),
            "avg_answer_relevance_graphrag": sum(r["answer_relevance_graphrag"] for r in rows) / len(rows),
            "details": rows,
        }

    out_path = settings.processed_data_dir / "evaluation.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    m.flush()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def cmd_quality_check(golden_path: Path, lang: str | None = None) -> dict:
    _llm().ensure_model_ready()
    rows = read_jsonl(golden_path)
    if not rows:
        raise RuntimeError(f"No golden rows in {golden_path}")

    tp = fp = fn = 0
    violations = 0

    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("quality_check"):
        for row in rows:
            text = str(row.get("text", ""))
            expected = row.get("relations", [])
            lang_active = _active_lang(lang, text_hint=text)

            predicted = extract_relations(_llm(), chunk_id=f"golden_{row.get('id','x')}", text=text, lang=lang_active)
            validated = validate_relations(predicted, min_confidence=0.5)
            violations += len(validated.invalid)

            pset = {(r.subject, r.predicate, r.object) for r in validated.valid}
            eset = {
                (
                    str(r.get("subject", "")).strip(),
                    str(r.get("predicate", "")).strip().upper(),
                    str(r.get("object", "")).strip(),
                )
                for r in expected
            }

            tp += len(pset & eset)
            fp += len(pset - eset)
            fn += len(eset - pset)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    result = {
        "total_cases": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "schema_violations": violations,
    }
    write_json(settings.quality_report_path, result)

    m.set_quality("relation_valid_rate", precision)
    m.flush()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def cmd_benchmark(cases_path: Path, top_k: int, lang: str | None = None) -> dict:
    _llm().ensure_model_ready()
    cases = _load_cases(cases_path)
    if not cases:
        raise RuntimeError(f"No benchmark cases in {cases_path}")

    rows = []
    m = _metrics()
    m.inc("operation_count", 1)
    with m.timed("benchmark"):
        for case in cases:
            case_lang = _active_lang(lang, text_hint=case.question)

            base_chunks, _ = _retrieve(case.question, top_k=top_k, use_graph=False)
            graph_chunks, graph_ctx = _retrieve(case.question, top_k=top_k, use_graph=True)

            base_ans = _safe_answer(
                case.question,
                base_chunks,
                graph_context=[],
                lang=case_lang,
                metrics=m,
            )
            graph_ans = _safe_answer(
                case.question,
                graph_chunks,
                graph_context=graph_ctx,
                lang=case_lang,
                metrics=m,
            )

            row = evaluate_question(case, base_ans, graph_ans, top_k)
            row["delta_precision"] = row["precision_at_k_graphrag"] - row["precision_at_k_baseline"]
            row["delta_relevance"] = row["answer_relevance_graphrag"] - row["answer_relevance_baseline"]
            rows.append(row)

    avg_delta_precision = sum(r["delta_precision"] for r in rows) / len(rows)
    avg_delta_relevance = sum(r["delta_relevance"] for r in rows) / len(rows)
    baseline = max(1e-9, sum(r["precision_at_k_baseline"] for r in rows) / len(rows))

    report = {
        "num_cases": len(rows),
        "avg_delta_precision": avg_delta_precision,
        "avg_delta_relevance": avg_delta_relevance,
        "relative_precision_gain": avg_delta_precision / baseline,
        "rows": rows,
    }

    write_json(settings.benchmark_report_path, report)
    m.flush()

    quality = read_json(settings.quality_report_path)
    _write_interview_report(quality if isinstance(quality, dict) else {}, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def cmd_retention_check(profile: str, apply: bool = False) -> dict:
    profile_payload = _profile(profile)
    retention_days = profile_payload.get("retention_days", {})
    roots = {
        "raw": settings.raw_data_dir,
        "processed": settings.processed_data_dir,
        "logs": Path("logs"),
        "eval": settings.eval_data_dir,
    }
    candidates = build_retention_candidates(retention_days, roots=roots)
    report = retention_report(candidates, settings.retention_report_path)
    if apply:
        report["apply_result"] = apply_retention(candidates)
        write_json(settings.retention_report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def cmd_health(mode: str) -> None:
    if mode == "live":
        result = health_live()
    else:
        result = health_ready(
            settings.ollama_base_url,
            settings.ollama_model,
            [settings.graph_pickle_path, settings.processed_data_dir / "chunks.jsonl"],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_preflight() -> None:
    settings.ensure_dirs()
    issues: list[str] = []
    info: dict[str, str] = {}

    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["ollama_base_url"] = settings.ollama_base_url
    info["ollama_model"] = settings.ollama_model
    info["default_language"] = settings.default_language
    info["app_env"] = settings.app_env

    if shutil.which("ollama") is None:
        issues.append("ollama binary not found in PATH")

    try:
        resp = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        if settings.ollama_model not in names:
            issues.append(f"model '{settings.ollama_model}' is not pulled")
    except Exception as exc:
        issues.append(f"cannot reach Ollama API: {exc}")

    print("\n=== Preflight ===")
    for k, v in info.items():
        print(f"{k}: {v}")

    if issues:
        print("\nStatus: FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("\nStatus: OK")


def cmd_full_run(
    n: int,
    question: str,
    top_k: int,
    eval_questions: Path,
    benchmark_cases: Path,
    lang: str | None = None,
) -> None:
    cmd_preflight()
    cmd_ingest_full(n=n, lang=lang)
    cmd_query(query=question, top_k=top_k, lang=lang)
    cmd_evaluate(questions_path=eval_questions, top_k=top_k, lang=lang)
    cmd_benchmark(cases_path=benchmark_cases, top_k=top_k, lang=lang)

    metrics = read_json(settings.ops_metrics_path)
    profile_payload = _profile(settings.app_env)
    alerts = evaluate_alerts(metrics if isinstance(metrics, dict) else {}, profile_payload.get("alerts", {}))
    write_json(settings.ops_alerts_path, {"alerts": alerts})


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GraphRAG CLI пайплайн (Interview-grade MVP+)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-data", help="Сгенерировать синтетические документы")
    p_gen.add_argument("--n", type=int, default=30)
    p_gen.add_argument("--lang", choices=["ru", "en"], default=None)

    p_build = sub.add_parser("build-index", help="Полная индексация latest-документов")

    p_ing_full = sub.add_parser("ingest-full", help="Full ingest: generate + governance + index")
    p_ing_full.add_argument("--n", type=int, default=30)
    p_ing_full.add_argument("--lang", choices=["ru", "en"], default=None)

    p_ing_inc = sub.add_parser("ingest-incremental", help="Incremental ingest только для new/updated")
    p_ing_inc.add_argument("--lang", choices=["ru", "en"], default=None)

    p_q = sub.add_parser("query", help="Выполнить GraphRAG-запрос")
    p_q.add_argument("query", type=str)
    p_q.add_argument("--top-k", type=int, default=settings.top_k)
    p_q.add_argument("--lang", choices=["ru", "en"], default=None)

    p_e = sub.add_parser("evaluate", help="Оценить baseline vs GraphRAG")
    p_e.add_argument("--questions", type=Path, default=settings.eval_data_dir / "questions.json")
    p_e.add_argument("--top-k", type=int, default=settings.top_k)
    p_e.add_argument("--lang", choices=["ru", "en"], default=None)

    p_qc = sub.add_parser("quality-check", help="Проверка extraction quality на golden-set")
    p_qc.add_argument("--golden", type=Path, default=settings.golden_dir / "golden_relations.jsonl")
    p_qc.add_argument("--lang", choices=["ru", "en"], default=None)

    p_bm = sub.add_parser("benchmark", help="Benchmark baseline vs GraphRAG")
    p_bm.add_argument("--cases", type=Path, default=settings.benchmark_dir / "cases.jsonl")
    p_bm.add_argument("--top-k", type=int, default=settings.top_k)
    p_bm.add_argument("--lang", choices=["ru", "en"], default=None)

    p_ret = sub.add_parser("retention-check", help="Проверка retention policy")
    p_ret.add_argument("--profile", type=str, default=settings.app_env)
    p_ret.add_argument("--apply", action="store_true")

    p_h = sub.add_parser("health", help="Health checks")
    p_h.add_argument("--mode", choices=["live", "ready"], default="ready")

    sub.add_parser("preflight", help="Проверить окружение и доступность Ollama")

    p_full = sub.add_parser("full-run", help="Полный запуск interview-grade пайплайна")
    p_full.add_argument("--n", type=int, default=30)
    p_full.add_argument("--question", type=str, default="Кто предложил решение проблемы дедлайна?")
    p_full.add_argument("--top-k", type=int, default=settings.top_k)
    p_full.add_argument("--questions", type=Path, default=settings.eval_data_dir / "questions.json")
    p_full.add_argument("--benchmark-cases", type=Path, default=settings.benchmark_dir / "cases.jsonl")
    p_full.add_argument("--lang", choices=["ru", "en"], default=None)

    return p


def main() -> None:
    run_id = setup_logging()
    logger.info("Логирование инициализировано: run_id=%s", run_id)
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "generate-data":
        cmd_generate_data(args.n, lang=args.lang)
    elif args.cmd == "build-index":
        cmd_build_index()
    elif args.cmd == "ingest-full":
        cmd_ingest_full(args.n, lang=args.lang)
    elif args.cmd == "ingest-incremental":
        cmd_ingest_incremental(lang=args.lang)
    elif args.cmd == "query":
        cmd_query(args.query, args.top_k, lang=args.lang)
    elif args.cmd == "evaluate":
        cmd_evaluate(args.questions, args.top_k, lang=args.lang)
    elif args.cmd == "quality-check":
        cmd_quality_check(args.golden, lang=args.lang)
    elif args.cmd == "benchmark":
        cmd_benchmark(args.cases, args.top_k, lang=args.lang)
    elif args.cmd == "retention-check":
        cmd_retention_check(args.profile, apply=args.apply)
    elif args.cmd == "health":
        cmd_health(args.mode)
    elif args.cmd == "preflight":
        cmd_preflight()
    elif args.cmd == "full-run":
        cmd_full_run(
            n=args.n,
            question=args.question,
            top_k=args.top_k,
            eval_questions=args.questions,
            benchmark_cases=args.benchmark_cases,
            lang=args.lang,
        )


if __name__ == "__main__":
    main()
