from pathlib import Path

from src.ingest.state import apply_versioning, latest_doc_ids, load_manifest
from src.models import DocumentRecord


def _doc(text: str) -> DocumentRecord:
    return DocumentRecord(
        id="draft",
        doc_type="chat",
        text=text,
        metadata={"external_id": "thread-42", "topic": "auth"},
    )


def test_apply_versioning_new_unchanged_updated(tmp_path: Path):
    manifest = load_manifest(path=tmp_path / "manifest.json")

    first, manifest, first_summary = apply_versioning([_doc("Иван предложил решение.")], manifest)
    assert first_summary["new"] == 1
    assert first[0].doc_version == 1
    first_id = first[0].id

    second, manifest, second_summary = apply_versioning([_doc("Иван предложил решение.")], manifest)
    assert second_summary["unchanged"] == 1
    assert second[0].id == first_id
    assert second[0].doc_version == 1

    third, manifest, third_summary = apply_versioning([_doc("Иван предложил и реализовал решение.")], manifest)
    assert third_summary["updated"] == 1
    assert third[0].doc_version == 2
    assert third[0].supersedes_doc_id == first_id

    active_ids = latest_doc_ids(manifest)
    assert third[0].id in active_ids
    assert first_id not in active_ids
