from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "90"))
    ollama_retries: int = int(os.getenv("OLLAMA_RETRIES", "3"))
    ollama_retry_backoff_sec: float = float(os.getenv("OLLAMA_RETRY_BACKOFF_SEC", "2.0"))
    default_language: str = os.getenv("DEFAULT_LANGUAGE", "ru")
    supported_languages: list[str] = field(
        default_factory=lambda: [
            x.strip() for x in os.getenv("SUPPORTED_LANGUAGES", "ru,en").split(",") if x.strip()
        ]
    )

    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    eval_data_dir: Path = Path(os.getenv("EVAL_DATA_DIR", "data/eval"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    synthetic_batch_size: int = int(os.getenv("SYNTHETIC_BATCH_SIZE", "6"))
    app_env: str = os.getenv("APP_ENV", "dev")

    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", ".chroma"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "graphrag_chunks")

    graph_pickle_path: Path = Path("data/processed/graph.gpickle")
    graph_json_path: Path = Path("data/processed/graph.json")
    documents_latest_path: Path = Path("data/processed/documents_latest.jsonl")
    governance_dir: Path = Path("data/governance")
    pii_mapping_path: Path = Path("data/governance/pii_mapping.jsonl")
    retention_report_path: Path = Path("data/governance/retention_report.json")
    ingest_dir: Path = Path("data/ingest")
    ingest_manifest_path: Path = Path("data/ingest/manifest.json")
    managed_documents_path: Path = Path("data/ingest/documents_managed.jsonl")
    golden_dir: Path = Path("data/golden")
    benchmark_dir: Path = Path("data/benchmark")
    ops_dir: Path = Path("data/ops")
    ops_metrics_path: Path = Path("data/ops/metrics.json")
    ops_metrics_prom_path: Path = Path("data/ops/metrics.prom")
    ops_alerts_path: Path = Path("data/ops/alerts.json")
    quality_report_path: Path = Path("data/processed/quality_report.json")
    benchmark_report_path: Path = Path("data/processed/benchmark_report.json")
    interview_report_path: Path = Path("data/processed/interview_report.json")
    ragas_report_path: Path = Path("data/processed/ragas_evaluation.json")
    profiles_dir: Path = Path("config/profiles")

    def ensure_dirs(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.governance_dir.mkdir(parents=True, exist_ok=True)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.ops_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
