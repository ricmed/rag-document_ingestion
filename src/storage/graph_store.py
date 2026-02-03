from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class Relationship:
    source_id: str
    target_id: str
    relation_type: str
    properties: dict[str, Any]


class GraphStore:
    """Persist and query entities/relationships in a lightweight graph."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    @property
    def entities(self) -> dict[str, Entity]:
        return dict(self._entities)

    @property
    def relationships(self) -> list[Relationship]:
        return list(self._relationships)

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity

    def add_relationship(self, relationship: Relationship) -> None:
        self._relationships.append(relationship)

    def save(self) -> None:
        if not self._storage_path:
            raise ValueError("storage_path is required to persist graph data")
        payload = {
            "entities": [asdict(entity) for entity in self._entities.values()],
            "relationships": [asdict(rel) for rel in self._relationships],
        }
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def load(self) -> None:
        if not self._storage_path:
            raise ValueError("storage_path is required to load graph data")
        if not self._storage_path.exists():
            self._entities = {}
            self._relationships = []
            return
        payload = json.loads(self._storage_path.read_text())
        self._entities = {
            item["entity_id"]: Entity(
                entity_id=item["entity_id"],
                entity_type=item["entity_type"],
                properties=dict(item["properties"]),
            )
            for item in payload.get("entities", [])
        }
        self._relationships = [
            Relationship(
                source_id=item["source_id"],
                target_id=item["target_id"],
                relation_type=item["relation_type"],
                properties=dict(item["properties"]),
            )
            for item in payload.get("relationships", [])
        ]

    def get_related_entities(
        self,
        entity_id: str,
        relationship_types: Iterable[str] | None = None,
        direction: str = "both",
    ) -> set[str]:
        related = set()
        rel_types = set(relationship_types) if relationship_types else None
        for rel in self._relationships:
            if rel_types and rel.relation_type not in rel_types:
                continue
            if direction in {"both", "out"} and rel.source_id == entity_id:
                related.add(rel.target_id)
            if direction in {"both", "in"} and rel.target_id == entity_id:
                related.add(rel.source_id)
        return related

    def query_entities(self, filters: dict[str, Any] | None = None) -> set[str] | None:
        if not filters:
            return None
        result_ids = set(self._entities.keys())
        entity_types = filters.get("entity_types")
        if entity_types:
            allowed_types = set(entity_types)
            result_ids = {
                entity_id
                for entity_id in result_ids
                if self._entities[entity_id].entity_type in allowed_types
            }
        related_to = filters.get("related_to")
        if related_to:
            relationship_types = filters.get("relationship_types")
            direction = filters.get("direction", "both")
            related = self.get_related_entities(
                related_to,
                relationship_types=relationship_types,
                direction=direction,
            )
            result_ids &= related
        return result_ids
