from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from src.io_utils import read_jsonl, write_jsonl


@dataclass
class PIIMatch:
    label: str
    value: str


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?:\+7|8)?[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")),
    ("PERSON", re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b")),
]


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def detect_pii(text: str) -> list[PIIMatch]:
    matches: list[PIIMatch] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if value:
                matches.append(PIIMatch(label=label, value=value))
    unique: dict[tuple[str, str], PIIMatch] = {(m.label, m.value): m for m in matches}
    return list(unique.values())


def _load_mapping(mapping_path: Path) -> list[dict]:
    return read_jsonl(mapping_path)


def _next_counter(rows: list[dict], label: str) -> int:
    nums = []
    for row in rows:
        if row.get("label") != label:
            continue
        pseudo = str(row.get("pseudonym", ""))
        parts = pseudo.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return (max(nums) + 1) if nums else 1


def pseudonymize_text(text: str, mapping_path: Path) -> tuple[str, list[dict]]:
    mapping = _load_mapping(mapping_path)
    by_hash = {row.get("raw_hash"): row for row in mapping}

    redacted = text
    created: list[dict] = []
    for match in detect_pii(text):
        raw_hash = _hash_value(match.value)
        row = by_hash.get(raw_hash)
        if row is None:
            counter = _next_counter(mapping + created, match.label)
            pseudonym = f"[{match.label}_{counter:04d}]"
            row = {
                "label": match.label,
                "raw_hash": raw_hash,
                "pseudonym": pseudonym,
            }
            created.append(row)
            by_hash[raw_hash] = row
        redacted = redacted.replace(match.value, row["pseudonym"])

    if created:
        write_jsonl(mapping_path, [*mapping, *created])
        try:
            mapping_path.chmod(0o600)
        except Exception:
            pass

    return redacted, created
