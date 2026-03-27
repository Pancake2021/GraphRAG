from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.io_utils import write_json


@dataclass
class RetentionCandidate:
    path: str
    artifact_type: str
    age_days: float
    ttl_days: int


def _iter_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return [p for p in base.rglob("*") if p.is_file()]


def build_retention_candidates(
    retention_days: dict[str, int],
    roots: dict[str, Path] | None = None,
) -> list[RetentionCandidate]:
    roots = roots or {
        "raw": Path("data/raw"),
        "processed": Path("data/processed"),
        "logs": Path("logs"),
        "eval": Path("data/eval"),
    }
    now = datetime.now(timezone.utc)
    out: list[RetentionCandidate] = []

    for typ, root in roots.items():
        ttl = int(retention_days.get(typ, 0))
        if ttl <= 0:
            continue
        for file_path in _iter_files(root):
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
            age = (now - mtime) / timedelta(days=1)
            if age > ttl:
                out.append(
                    RetentionCandidate(
                        path=str(file_path),
                        artifact_type=typ,
                        age_days=round(float(age), 2),
                        ttl_days=ttl,
                    )
                )
    return out


def retention_report(candidates: list[RetentionCandidate], report_path: Path) -> dict:
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [c.__dict__ for c in candidates],
        "total_candidates": len(candidates),
    }
    write_json(report_path, payload)
    return payload


def apply_retention(candidates: list[RetentionCandidate]) -> dict:
    removed = 0
    errors: list[str] = []
    for c in candidates:
        try:
            Path(c.path).unlink(missing_ok=True)
            removed += 1
        except Exception as exc:
            errors.append(f"{c.path}: {exc}")
    return {"removed": removed, "errors": errors}
