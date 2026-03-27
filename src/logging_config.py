from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _ensure_logs_dir() -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def setup_logging(level: int = logging.INFO) -> str:
    logs_dir = _ensure_logs_dir()
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log_path = logs_dir / f"run-{ts}-{run_id}.log"
    latest_log_path = logs_dir / "latest.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | run=%(run_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(lambda record: setattr(record, "run_id", run_id) or True)
    root.addHandler(console)

    run_file = RotatingFileHandler(run_log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    run_file.setFormatter(fmt)
    run_file.addFilter(lambda record: setattr(record, "run_id", run_id) or True)
    root.addHandler(run_file)

    latest_file = RotatingFileHandler(latest_log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    latest_file.setFormatter(fmt)
    latest_file.addFilter(lambda record: setattr(record, "run_id", run_id) or True)
    root.addHandler(latest_file)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return run_id

