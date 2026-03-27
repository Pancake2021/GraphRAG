from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.io_utils import read_json, write_json
from src.models import DocumentRecord
from src.preprocessing.chunker import normalize_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def doc_fingerprint(doc_type: str, text: str, metadata: dict) -> str:
    normalized = normalize_text(text)
    key = f"{doc_type}|{normalized}|{sorted(metadata.items())}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def external_id_for_doc(doc: DocumentRecord) -> str:
    ext = str(doc.metadata.get("external_id", "")).strip()
    if ext:
        return ext
    topic = str(doc.metadata.get("topic", "")).strip()
    attendees = "|".join(sorted([str(x) for x in doc.metadata.get("attendees", [])]))
    seed = f"{doc.doc_type}|{topic}|{attendees}|{normalize_text(doc.text)[:120]}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def load_manifest(path: Path) -> dict:
    payload = read_json(path)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("last_run_at", None)
    payload.setdefault("documents", {})
    return payload


def save_manifest(path: Path, manifest: dict) -> None:
    manifest["last_run_at"] = _now()
    write_json(path, manifest)


def apply_versioning(docs: list[DocumentRecord], manifest: dict) -> tuple[list[DocumentRecord], dict, dict]:
    docs_state: dict = manifest.setdefault("documents", {})

    processed: list[DocumentRecord] = []
    summary = {"new": 0, "unchanged": 0, "updated": 0, "invalid": 0}

    for doc in docs:
        if not doc.text.strip():
            doc.ingest_status = "invalid"
            summary["invalid"] += 1
            processed.append(doc)
            continue

        ext_id = external_id_for_doc(doc)
        fp = doc_fingerprint(doc.doc_type, doc.text, doc.metadata)

        state = docs_state.get(ext_id)
        if state is None:
            version = 1
            doc_id = f"{ext_id}_v{version}"
            doc.id = doc_id
            doc.doc_version = version
            doc.doc_fingerprint = fp
            doc.is_latest = True
            doc.supersedes_doc_id = None
            doc.ingest_status = "new"
            summary["new"] += 1

            docs_state[ext_id] = {
                "external_id": ext_id,
                "latest_doc_id": doc_id,
                "latest_version": version,
                "latest_fingerprint": fp,
                "history": [
                    {
                        "doc_id": doc_id,
                        "version": version,
                        "fingerprint": fp,
                        "status": "new",
                        "timestamp": _now(),
                    }
                ],
            }
            processed.append(doc)
            continue

        prev_fp = state.get("latest_fingerprint")
        prev_doc_id = state.get("latest_doc_id")
        prev_version = int(state.get("latest_version", 1))

        if prev_fp == fp:
            doc.id = prev_doc_id
            doc.doc_version = prev_version
            doc.doc_fingerprint = fp
            doc.is_latest = True
            doc.supersedes_doc_id = None
            doc.ingest_status = "unchanged"
            summary["unchanged"] += 1
            processed.append(doc)
            continue

        version = prev_version + 1
        doc_id = f"{ext_id}_v{version}"
        doc.id = doc_id
        doc.doc_version = version
        doc.doc_fingerprint = fp
        doc.is_latest = True
        doc.supersedes_doc_id = prev_doc_id
        doc.ingest_status = "updated"
        summary["updated"] += 1

        state["latest_doc_id"] = doc_id
        state["latest_version"] = version
        state["latest_fingerprint"] = fp
        state.setdefault("history", []).append(
            {
                "doc_id": doc_id,
                "version": version,
                "fingerprint": fp,
                "status": "updated",
                "supersedes": prev_doc_id,
                "timestamp": _now(),
            }
        )
        processed.append(doc)

    return processed, manifest, summary


def latest_doc_ids(manifest: dict) -> set[str]:
    docs = manifest.get("documents", {})
    return {v.get("latest_doc_id") for v in docs.values() if v.get("latest_doc_id")}
