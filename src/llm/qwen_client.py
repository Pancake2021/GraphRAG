from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests


class QwenClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 90,
        retries: int = 3,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)
        self.logger = logging.getLogger(__name__)

    def ensure_model_ready(self) -> None:
        url = f"{self.base_url}/api/tags"
        self.logger.info("Checking Ollama model availability: model=%s url=%s", self.model, url)
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        names = {m.get("name", "") for m in payload.get("models", [])}
        if self.model not in names:
            raise RuntimeError(
                f"Model '{self.model}' not found in Ollama. Pull it with: ollama pull {self.model}"
            )

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/generate"
        full_prompt = prompt if system is None else f"System:\n{system}\n\nUser:\n{prompt}"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self.logger.info(
                    "LLM request started: model=%s attempt=%s/%s timeout=%ss",
                    self.model,
                    attempt,
                    self.retries,
                    self.timeout,
                )
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                self.logger.info("LLM request succeeded: model=%s attempt=%s", self.model, attempt)
                return data.get("response", "").strip()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
                last_exc = exc
                self.logger.warning(
                    "LLM timeout: model=%s attempt=%s/%s timeout=%ss err=%s",
                    self.model,
                    attempt,
                    self.retries,
                    self.timeout,
                    exc,
                )
                if attempt >= self.retries:
                    break
                sleep_sec = self.retry_backoff_sec * attempt
                self.logger.info("Sleeping before retry: %.1fs", sleep_sec)
                time.sleep(sleep_sec)
        raise RuntimeError(
            f"Ollama request timed out after {self.retries} attempts "
            f"(timeout={self.timeout}s). Reduce request size or increase OLLAMA_TIMEOUT."
        ) from last_exc

    def generate_json(
        self, prompt: str, system: str | None = None, temperature: float = 0.0
    ) -> dict[str, Any]:
        text = self.generate(prompt, system=system, temperature=temperature)
        parsed = self._parse_json_candidates(text)
        if parsed is not None:
            return parsed

        # One repair attempt: ask model to rewrite exactly valid JSON.
        repair_prompt = (
            "Исправь и верни строго валидный JSON без markdown и пояснений.\n\n"
            if "{" in text
            else "Верни строго валидный JSON без markdown и пояснений.\n\n"
        ) + text
        repaired = self.generate(repair_prompt, system=system, temperature=0.0)
        parsed_repaired = self._parse_json_candidates(repaired)
        if parsed_repaired is not None:
            return parsed_repaired
        raise ValueError(f"LLM returned invalid JSON after repair attempt: {repaired[:400]}")

    def _parse_json_candidates(self, text: str) -> dict[str, Any] | None:
        for candidate in self._json_candidates(text):
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                continue
        return None

    def _json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        # ```json ... ```
        for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE):
            candidates.append(m.group(1).strip())

        # Balanced JSON object slices
        starts = [i for i, ch in enumerate(text) if ch == "{"]
        for start in starts:
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1].strip())
                        break

        # Original text as last resort
        candidates.append(text.strip())

        # Keep order but deduplicate
        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered
