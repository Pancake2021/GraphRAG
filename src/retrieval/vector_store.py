from __future__ import annotations

import hashlib

from src.models import ChunkRecord


class _SimpleEmbeddingFn:
    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [int(b) / 255.0 for b in h[:64]]
            vectors.append(vec)
        return vectors


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            import chromadb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "chromadb is not installed. Install dependencies with `uv pip install -e .`"
            ) from exc
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = self._load_embedding_fn()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )
        self.collection_name = collection_name

    def _load_embedding_fn(self):
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            return SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            return _SimpleEmbeddingFn()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metas = []
        for c in chunks:
            ents = sorted({v for vals in c.entities.values() for v in vals})
            metas.append(
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.id,
                    "entities": "|".join(ents),
                    "doc_type": str(c.metadata.get("doc_type", "")),
                }
            )
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float, dict]]:
        result = self.collection.query(query_texts=[query], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
        metas = result.get("metadatas", [[]])[0] if result.get("metadatas") else [{} for _ in ids]
        rows: list[tuple[str, float, dict]] = []
        for chunk_id, dist, meta in zip(ids, distances, metas):
            score = 1.0 / (1.0 + float(dist))
            rows.append((chunk_id, score, meta))
        return rows

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    def reset_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )
