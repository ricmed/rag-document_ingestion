"""Audio transcription and ingestion helpers.

This module provides a configurable transcription layer (local Whisper or API-based)
plus time-based segmentation and hooks to reuse an existing chunking/embedding
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class AudioSegment:
    """A transcribed segment with time boundaries in seconds."""

    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuration for transcription backends."""

    provider: str = "whisper_local"
    model: str = "base"
    language: Optional[str] = None


class AudioTranscriber(Protocol):
    """Protocol for audio transcribers."""

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> List[AudioSegment]:
        """Return time-stamped segments for the provided audio file."""


class ChunkEmbeddingPipeline(Protocol):
    """Protocol for chunking/embedding pipeline integration."""

    def process(self, text: str, metadata: dict) -> object:
        """Process a text chunk and return an embedding/artifact."""


class WhisperLocalTranscriber:
    """Local Whisper transcription using the `whisper` Python package."""

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> List[AudioSegment]:
        try:
            import whisper  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "Local Whisper transcriber requires the `whisper` package. "
                "Install with `pip install openai-whisper`."
            ) from exc

        model = whisper.load_model(config.model)
        result = model.transcribe(str(audio_path), language=config.language)
        segments = result.get("segments", [])
        return [
            AudioSegment(start_s=segment["start"], end_s=segment["end"], text=segment["text"].strip())
            for segment in segments
        ]


class OpenAIAPIAudioTranscriber:
    """OpenAI API transcription via the `openai` Python client."""

    def __init__(self, client: Optional[object] = None) -> None:
        self._client = client

    def transcribe(self, audio_path: Path, config: TranscriptionConfig) -> List[AudioSegment]:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError(
                "OpenAI API transcriber requires the `openai` package. "
                "Install with `pip install openai`."
            ) from exc

        client = self._client or OpenAI()
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model=config.model,
                language=config.language,
                response_format="verbose_json",
            )

        segments = getattr(response, "segments", None) or []
        return [
            AudioSegment(start_s=segment.start, end_s=segment.end, text=segment.text.strip())
            for segment in segments
        ]


def chunk_segments_by_time(
    segments: Sequence[AudioSegment],
    max_duration_s: float = 60.0,
    min_duration_s: float = 10.0,
) -> List[AudioSegment]:
    """Group segments into chunks with a max duration.

    Returns new segments whose time window is derived from the grouped segments.
    """

    if not segments:
        return []

    chunks: List[AudioSegment] = []
    current_text: List[str] = []
    chunk_start = segments[0].start_s
    chunk_end = segments[0].end_s

    def flush() -> None:
        nonlocal current_text, chunk_start, chunk_end
        if not current_text:
            return
        chunks.append(
            AudioSegment(
                start_s=chunk_start,
                end_s=chunk_end,
                text=" ".join(current_text).strip(),
            )
        )
        current_text = []

    for segment in segments:
        if not current_text:
            chunk_start = segment.start_s
            chunk_end = segment.end_s
            current_text.append(segment.text)
            continue

        projected_end = segment.end_s
        projected_duration = projected_end - chunk_start
        if projected_duration <= max_duration_s:
            current_text.append(segment.text)
            chunk_end = segment.end_s
        else:
            flush()
            chunk_start = segment.start_s
            chunk_end = segment.end_s
            current_text.append(segment.text)

    flush()

    if min_duration_s <= 0:
        return chunks

    merged: List[AudioSegment] = []
    buffer: Optional[AudioSegment] = None
    for chunk in chunks:
        if buffer is None:
            buffer = chunk
            continue
        if (buffer.end_s - buffer.start_s) < min_duration_s:
            buffer = AudioSegment(
                start_s=buffer.start_s,
                end_s=chunk.end_s,
                text=f"{buffer.text} {chunk.text}".strip(),
            )
        else:
            merged.append(buffer)
            buffer = chunk

    if buffer is not None:
        merged.append(buffer)

    return merged


def process_audio_for_ingestion(
    audio_path: Path,
    pipeline: ChunkEmbeddingPipeline,
    transcriber: AudioTranscriber,
    config: Optional[TranscriptionConfig] = None,
    max_duration_s: float = 60.0,
    min_duration_s: float = 10.0,
) -> List[object]:
    """Transcribe audio, chunk by time, and send to chunking/embedding pipeline."""

    config = config or TranscriptionConfig()
    segments = transcriber.transcribe(audio_path, config)
    chunks = chunk_segments_by_time(
        segments, max_duration_s=max_duration_s, min_duration_s=min_duration_s
    )

    results: List[object] = []
    for chunk in chunks:
        metadata = {
            "source": str(audio_path),
            "start_s": chunk.start_s,
            "end_s": chunk.end_s,
        }
        results.append(pipeline.process(chunk.text, metadata))

    return results


def get_transcriber(config: TranscriptionConfig) -> AudioTranscriber:
    """Factory for transcribers based on config."""

    if config.provider == "whisper_local":
        return WhisperLocalTranscriber()
    if config.provider == "openai":
        return OpenAIAPIAudioTranscriber()
    raise ValueError(f"Unsupported transcription provider: {config.provider}")
