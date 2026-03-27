from __future__ import annotations

from pathlib import Path

import requests


def health_live() -> dict:
    return {"status": "ok", "mode": "live"}


def health_ready(ollama_base_url: str, model: str, required_paths: list[Path]) -> dict:
    checks: dict[str, str] = {}
    ok = True

    try:
        resp = requests.get(f"{ollama_base_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        checks["ollama"] = "ok" if model in names else f"model_missing:{model}"
        ok = ok and (model in names)
    except Exception as exc:
        checks["ollama"] = f"error:{exc}"
        ok = False

    for p in required_paths:
        exists = p.exists()
        checks[f"path:{p}"] = "ok" if exists else "missing"
        ok = ok and exists

    return {"status": "ok" if ok else "failed", "mode": "ready", "checks": checks}
