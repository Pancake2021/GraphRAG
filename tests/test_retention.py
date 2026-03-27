from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.governance.retention import build_retention_candidates


def _set_mtime(path: Path, dt: datetime) -> None:
    ts = dt.timestamp()
    path.touch()
    path.chmod(0o600)
    import os

    os.utime(path, (ts, ts))


def test_retention_candidates_with_custom_roots(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    logs_dir = tmp_path / "logs"
    eval_dir = tmp_path / "eval"
    for d in (raw_dir, proc_dir, logs_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    old_file = raw_dir / "old.jsonl"
    fresh_file = raw_dir / "fresh.jsonl"
    _set_mtime(old_file, now - timedelta(days=40))
    _set_mtime(fresh_file, now - timedelta(days=1))

    candidates = build_retention_candidates(
        retention_days={"raw": 30, "processed": 30, "logs": 30, "eval": 30},
        roots={"raw": raw_dir, "processed": proc_dir, "logs": logs_dir, "eval": eval_dir},
    )

    paths = {c.path for c in candidates}
    assert str(old_file) in paths
    assert str(fresh_file) not in paths

