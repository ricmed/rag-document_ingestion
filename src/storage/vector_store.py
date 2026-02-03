from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .graph_store import GraphStore


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    embedding: list[float]
    metadata: dict[str, Any]


class VectorStore:
    """Persist and query vector embeddings with metadata."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._records: list[VectorRecord] = []

    @property
    def records(self) -> list[VectorRecord]:
        return list(self._records)

    def add_record(self, record: VectorRecord) -> None:
        self._records.append(record)

    def save(self) -> None:
        if not self._storage_path:
            raise ValueError("storage_path is required to persist records")
        payload = [asdict(record) for record in self._records]
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self) -> None:
        if not self._storage_path:
            raise ValueError("storage_path is required to load records")
        if not self._storage_path.exists():
            self._records = []
            return
        payload = json.loads(self._storage_path.read_text())
        self._records = [
            VectorRecord(
                record_id=item["record_id"],
                embedding=list(item["embedding"]),
                metadata=dict(item["metadata"]),
            )
            for item in payload
        ]

    def query(
        self,
        embedding: Iterable[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = list(embedding)
        scored = []
        for record in self._records:
            if not _metadata_match(record.metadata, filters):
                continue
            score = _cosine_similarity(query_embedding, record.embedding)
            scored.append({"record": record, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def combined_query(
        self,
        embedding: Iterable[float],
        graph_store: GraphStore,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
        relation_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_entity_ids = graph_store.query_entities(relation_filters)
        combined_filters = dict(metadata_filters or {})
        if allowed_entity_ids is not None:
            combined_filters["entity_id"] = allowed_entity_ids
        return self.query(embedding, top_k=top_k, filters=combined_filters)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _metadata_match(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        value = metadata.get(key)
        if isinstance(expected, (set, list, tuple)):
            if value not in expected:
                return False
        else:
            if value != expected:
                return False
    return True
