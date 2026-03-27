from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PredicateType = Literal[
    "PROPOSED",
    "SOLVED",
    "PARTICIPATED_IN",
    "RELATED_TO",
    "MENTIONED_IN",
]

NodeType = Literal[
    "Person",
    "Problem",
    "Solution",
    "Decision",
    "Meeting",
    "Document",
    "Organization",
    "Date",
]


class DocumentRecord(BaseModel):
    id: str
    doc_type: Literal["chat", "meeting", "brainstorm", "generic"]
    text: str
    source_text_hash: str | None = None
    doc_fingerprint: str | None = None
    doc_version: int = 1
    is_latest: bool = True
    supersedes_doc_id: str | None = None
    ingest_status: Literal["new", "unchanged", "updated", "invalid"] = "new"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    subject: str
    predicate: PredicateType
    object: str
    confidence: float = 0.8
    source_chunk_id: str


class ChunkRecord(BaseModel):
    id: str
    doc_id: str
    text: str
    entities: dict[str, list[str]] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerResult(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    graph_context: list[dict[str, Any]]
    retrieved_chunks: list[ChunkRecord]


class EvalQuestion(BaseModel):
    id: str
    question: str
    ground_truth: str
