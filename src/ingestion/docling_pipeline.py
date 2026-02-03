"""Docling-based ingestion pipeline for PDFs and Markdown."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol


class ImageDescriber(Protocol):
    """Interface for describing images using LLMs or external services."""

    def describe(self, image_bytes: bytes, metadata: Mapping[str, Any] | None = None) -> str:
        ...


class DocumentStore(Protocol):
    """Interface for persisting enriched documents for indexing."""

    def write(self, documents: Iterable["EnrichedDocument"]) -> None:
        ...


@dataclass
class EnrichedImage:
    image_id: str
    description: str
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    data_base64: str | None = None


@dataclass
class EnrichedDocument:
    source_path: str
    metadata: dict[str, Any]
    text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    code_blocks: list[dict[str, Any]] = field(default_factory=list)
    images: list[EnrichedImage] = field(default_factory=list)
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class FileDocumentStore:
    """Persist enriched documents as JSONL on disk."""

    output_path: Path

    def write(self, documents: Iterable[EnrichedDocument]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(_to_serializable(document), ensure_ascii=False))
                handle.write("\n")


@dataclass
class DoclingPipeline:
    """Ingestion pipeline built on top of Docling."""

    image_describer: ImageDescriber
    document_store: DocumentStore
    converter_factory: Callable[[], Any] | None = None

    def ingest(self, paths: Iterable[str | Path]) -> list[EnrichedDocument]:
        converter = self._create_converter()
        enriched_docs: list[EnrichedDocument] = []
        for path in paths:
            enriched_docs.append(self._ingest_path(converter, Path(path)))
        self.document_store.write(enriched_docs)
        return enriched_docs

    def _create_converter(self) -> Any:
        if self.converter_factory is not None:
            return self.converter_factory()
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Docling is not installed. Install `docling` to use DoclingPipeline."
            ) from exc
        return DocumentConverter()

    def _ingest_path(self, converter: Any, path: Path) -> EnrichedDocument:
        conversion_result = converter.convert(path)
        document = conversion_result.document
        doc_dict = _export_docling_dict(document)
        metadata = _extract_metadata(doc_dict, path)
        text = _extract_text(doc_dict)
        tables = _extract_tables(doc_dict)
        code_blocks = _extract_code_blocks(doc_dict)
        images = _extract_images(doc_dict, self.image_describer)
        return EnrichedDocument(
            source_path=str(path),
            metadata=metadata,
            text=text,
            tables=tables,
            code_blocks=code_blocks,
            images=images,
        )


def _export_docling_dict(document: Any) -> dict[str, Any]:
    for method_name in ("export_to_dict", "to_dict", "export_to_json"):
        method = getattr(document, method_name, None)
        if callable(method):
            exported = method()
            if isinstance(exported, str):
                return json.loads(exported)
            if isinstance(exported, dict):
                return exported
    raise RuntimeError("Unsupported Docling document export format.")


def _extract_metadata(doc_dict: Mapping[str, Any], path: Path) -> dict[str, Any]:
    metadata = dict(doc_dict.get("metadata", {}))
    metadata.setdefault("source_name", path.name)
    metadata.setdefault("source_type", path.suffix.lstrip(".").lower())
    metadata.setdefault("source_path", str(path))
    return metadata


def _extract_text(doc_dict: Mapping[str, Any]) -> str:
    if "text" in doc_dict:
        return str(doc_dict["text"])
    blocks = []
    for block in doc_dict.get("blocks", []):
        if block.get("type") in {"paragraph", "text", "heading"}:
            content = block.get("text") or block.get("content")
            if content:
                blocks.append(str(content))
    return "\n\n".join(blocks)


def _extract_tables(doc_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    tables = list(doc_dict.get("tables", []))
    for block in doc_dict.get("blocks", []):
        if block.get("type") == "table":
            tables.append(block)
    return tables


def _extract_code_blocks(doc_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    code_blocks = list(doc_dict.get("code_blocks", []))
    for block in doc_dict.get("blocks", []):
        if block.get("type") == "code":
            code_blocks.append(block)
    return code_blocks


def _extract_images(
    doc_dict: Mapping[str, Any], image_describer: ImageDescriber
) -> list[EnrichedImage]:
    images: list[EnrichedImage] = []
    for image in _iter_images(doc_dict):
        image_bytes = _decode_image_bytes(image)
        description = ""
        if image_bytes:
            description = image_describer.describe(image_bytes, image)
        images.append(
            EnrichedImage(
                image_id=str(image.get("id") or image.get("name") or ""),
                description=description,
                mime_type=image.get("mime_type"),
                width=image.get("width"),
                height=image.get("height"),
                data_base64=base64.b64encode(image_bytes).decode("utf-8")
                if image_bytes
                else None,
            )
        )
    return images


def _iter_images(doc_dict: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for image in doc_dict.get("images", []):
        yield image
    for block in doc_dict.get("blocks", []):
        if block.get("type") == "image":
            yield block


def _decode_image_bytes(image: Mapping[str, Any]) -> bytes | None:
    if "bytes" in image and isinstance(image["bytes"], (bytes, bytearray)):
        return bytes(image["bytes"])
    if "data" in image and isinstance(image["data"], str):
        try:
            return base64.b64decode(image["data"].encode("utf-8"))
        except (ValueError, TypeError):
            return None
    return None


def _to_serializable(document: EnrichedDocument) -> dict[str, Any]:
    return {
        "source_path": document.source_path,
        "metadata": document.metadata,
        "text": document.text,
        "tables": document.tables,
        "code_blocks": document.code_blocks,
        "images": [image.__dict__ for image in document.images],
        "extracted_at": document.extracted_at,
    }
