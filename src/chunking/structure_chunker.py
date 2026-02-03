"""Utilities for chunking markdown-like documents by structure.

Primary delimiters are headings/sections, with code/table blocks preserved.
Token limits are enforced with late chunking when necessary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


HEADING_RE = re.compile(r"^\s*#{1,6}\s+.+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*[:\-]+(?:\s*\|\s*[:\-]+)+\s*\|?\s*$")


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 256
    late_chunking: bool = True
    include_title_in_chunks: bool = True

    @staticmethod
    def from_json(path: Path) -> "ChunkingConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChunkingConfig(
            max_tokens=int(data.get("max_tokens", 256)),
            late_chunking=bool(data.get("late_chunking", True)),
            include_title_in_chunks=bool(data.get("include_title_in_chunks", True)),
        )


@dataclass
class Block:
    type: str
    text: str


@dataclass
class Section:
    title: Optional[str]
    blocks: List[Block]


class StructureChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def chunk(self, text: str) -> List[str]:
        blocks = list(self._extract_blocks(text))
        sections = self._split_sections(blocks)
        chunks: List[str] = []
        for section in sections:
            chunks.extend(self._chunk_section(section))
        return chunks

    def _extract_blocks(self, text: str) -> Iterable[Block]:
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                fence = line.strip()[:3]
                block_lines = [line]
                i += 1
                while i < len(lines):
                    block_lines.append(lines[i])
                    if lines[i].strip().startswith(fence):
                        i += 1
                        break
                    i += 1
                yield Block("code", "\n".join(block_lines))
                continue

            if self._is_table_start(lines, i):
                block_lines = [lines[i], lines[i + 1]]
                i += 2
                while i < len(lines) and self._is_table_line(lines[i]):
                    block_lines.append(lines[i])
                    i += 1
                yield Block("table", "\n".join(block_lines))
                continue

            if HEADING_RE.match(line):
                yield Block("heading", line)
                i += 1
                continue

            paragraph_lines = [line]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (
                    HEADING_RE.match(next_line)
                    or next_line.strip().startswith("```")
                    or self._is_table_start(lines, i)
                ):
                    break
                paragraph_lines.append(next_line)
                i += 1
            yield Block("text", "\n".join(paragraph_lines))

    def _split_sections(self, blocks: List[Block]) -> List[Section]:
        sections: List[Section] = []
        current: Optional[Section] = None
        for block in blocks:
            if block.type == "heading":
                if current:
                    sections.append(current)
                current = Section(title=block.text.strip(), blocks=[block])
            else:
                if current is None:
                    current = Section(title=None, blocks=[])
                current.blocks.append(block)
        if current:
            sections.append(current)
        return sections

    def _chunk_section(self, section: Section) -> List[str]:
        max_tokens = self.config.max_tokens
        blocks = section.blocks
        chunks: List[str] = []
        current_blocks: List[str] = []
        current_tokens = 0
        title_text = section.title or ""

        def flush() -> None:
            nonlocal current_blocks, current_tokens
            if current_blocks:
                chunks.append("\n".join(current_blocks).strip())
            current_blocks = []
            current_tokens = 0

        if self.config.include_title_in_chunks and title_text:
            title_tokens = self._count_tokens(title_text)
        else:
            title_tokens = 0

        for block in blocks:
            block_text = block.text
            if block.type == "heading" and not self.config.include_title_in_chunks:
                continue
            if block.type == "heading" and self.config.include_title_in_chunks:
                if current_blocks:
                    flush()
                current_blocks.append(block_text)
                current_tokens = self._count_tokens(block_text)
                continue

            block_tokens = self._count_tokens(block_text)
            if current_tokens + block_tokens <= max_tokens:
                current_blocks.append(block_text)
                current_tokens += block_tokens
                continue

            if current_blocks:
                flush()

            if block_tokens <= max_tokens:
                current_blocks.append(block_text)
                current_tokens = block_tokens
                continue

            if block.type in {"code", "table"}:
                current_blocks.append(block_text)
                current_tokens = block_tokens
                flush()
                continue

            if self.config.late_chunking:
                for piece in self._late_chunk_text(block_text, max_tokens, title_text if title_text else None, title_tokens):
                    chunks.append(piece)
                continue

            chunks.extend(self._hard_split(block_text, max_tokens))

        if current_blocks:
            flush()
        return chunks

    def _late_chunk_text(
        self,
        text: str,
        max_tokens: int,
        title_text: Optional[str],
        title_tokens: int,
    ) -> List[str]:
        chunks: List[str] = []
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        current: List[str] = []
        current_tokens = 0
        base_tokens = title_tokens if self.config.include_title_in_chunks and title_text else 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                chunk_text = "\n\n".join(current).strip()
                if title_text and self.config.include_title_in_chunks:
                    chunk_text = f"{title_text}\n{chunk_text}"
                chunks.append(chunk_text)
            current = []
            current_tokens = 0

        for paragraph in paragraphs:
            para_tokens = self._count_tokens(paragraph)
            if current_tokens + para_tokens + base_tokens <= max_tokens:
                current.append(paragraph)
                current_tokens += para_tokens
                continue

            if current:
                flush()

            if para_tokens + base_tokens <= max_tokens:
                current.append(paragraph)
                current_tokens = para_tokens
                continue

            for sentence in self._split_sentences(paragraph):
                sentence_tokens = self._count_tokens(sentence)
                if current_tokens + sentence_tokens + base_tokens <= max_tokens:
                    current.append(sentence)
                    current_tokens += sentence_tokens
                    continue
                if current:
                    flush()
                if sentence_tokens + base_tokens <= max_tokens:
                    current.append(sentence)
                    current_tokens = sentence_tokens
                    continue
                chunks.extend(self._hard_split(sentence, max_tokens, title_text))
        if current:
            flush()
        return chunks

    def _hard_split(self, text: str, max_tokens: int, title_text: Optional[str] = None) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        count = 0
        base = 0
        if title_text and self.config.include_title_in_chunks:
            base = self._count_tokens(title_text)
        for word in words:
            word_tokens = 1
            if count + word_tokens + base <= max_tokens:
                current.append(word)
                count += word_tokens
                continue
            chunk_text = " ".join(current).strip()
            if title_text and self.config.include_title_in_chunks:
                chunk_text = f"{title_text}\n{chunk_text}"
            chunks.append(chunk_text)
            current = [word]
            count = word_tokens
        if current:
            chunk_text = " ".join(current).strip()
            if title_text and self.config.include_title_in_chunks:
                chunk_text = f"{title_text}\n{chunk_text}"
            chunks.append(chunk_text)
        return chunks

    @staticmethod
    def _split_sentences(paragraph: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁ-Ú])", paragraph)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _is_table_start(lines: List[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False
        return StructureChunker._is_table_line(lines[index]) and TABLE_SEPARATOR_RE.match(lines[index + 1]) is not None

    @staticmethod
    def _is_table_line(line: str) -> bool:
        return "|" in line


def load_chunker(config_path: str | Path) -> StructureChunker:
    config = ChunkingConfig.from_json(Path(config_path))
    return StructureChunker(config)
