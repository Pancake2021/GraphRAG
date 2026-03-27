from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.io_utils import read_json, write_json


class MetricsStore:
    def __init__(self, metrics_path: Path, prom_path: Path) -> None:
        self.metrics_path = metrics_path
        self.prom_path = prom_path
        self.data = self._load()

    def _load(self) -> dict:
        payload = read_json(self.metrics_path)
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("counters", {})
        payload.setdefault("latency", {})
        payload.setdefault("quality", {})
        payload.setdefault("updated_at", None)
        return payload

    def inc(self, name: str, value: int = 1) -> None:
        self.data["counters"][name] = int(self.data["counters"].get(name, 0)) + value

    def set_quality(self, name: str, value: float) -> None:
        self.data["quality"][name] = float(value)

    def observe_latency(self, stage: str, seconds: float) -> None:
        hist = self.data["latency"].setdefault(stage, {"count": 0, "sum": 0.0, "max": 0.0})
        hist["count"] += 1
        hist["sum"] += float(seconds)
        hist["max"] = max(float(hist["max"]), float(seconds))

    @contextmanager
    def timed(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.inc("error_count", 1)
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.observe_latency(stage, elapsed)

    def flush(self) -> None:
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.metrics_path, self.data)
        self.prom_path.parent.mkdir(parents=True, exist_ok=True)
        self.prom_path.write_text(self._as_prometheus(), encoding="utf-8")

    def _as_prometheus(self) -> str:
        lines: list[str] = []
        for k, v in self.data.get("counters", {}).items():
            lines.append(f"graphrag_{k} {v}")
        for stage, vals in self.data.get("latency", {}).items():
            lines.append(f"graphrag_latency_count{{stage=\"{stage}\"}} {vals.get('count', 0)}")
            lines.append(f"graphrag_latency_sum{{stage=\"{stage}\"}} {vals.get('sum', 0.0)}")
            lines.append(f"graphrag_latency_max{{stage=\"{stage}\"}} {vals.get('max', 0.0)}")
        for k, v in self.data.get("quality", {}).items():
            lines.append(f"graphrag_quality{{name=\"{k}\"}} {v}")
        return "\n".join(lines) + "\n"


def evaluate_alerts(metrics: dict, thresholds: dict) -> list[dict]:
    alerts: list[dict] = []
    max_stage_latency = float(thresholds.get("max_stage_latency_sec", 30))
    min_relation_valid = float(thresholds.get("min_relation_valid_rate", 0.7))
    max_error_rate = float(thresholds.get("max_error_rate", 0.2))

    for stage, vals in metrics.get("latency", {}).items():
        if float(vals.get("max", 0.0)) > max_stage_latency:
            alerts.append({"type": "latency", "stage": stage, "value": vals.get("max"), "threshold": max_stage_latency})

    relation_valid = float(metrics.get("quality", {}).get("relation_valid_rate", 1.0))
    if relation_valid < min_relation_valid:
        alerts.append({"type": "quality", "name": "relation_valid_rate", "value": relation_valid, "threshold": min_relation_valid})

    counters = metrics.get("counters", {})
    total_ops = max(1, int(counters.get("operation_count", 1)))
    error_rate = float(counters.get("error_count", 0)) / float(total_ops)
    if error_rate > max_error_rate:
        alerts.append({"type": "error_rate", "value": error_rate, "threshold": max_error_rate})

    return alerts
