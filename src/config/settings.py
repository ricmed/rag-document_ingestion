"""Aplicação de configuração e seleção automática de device."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import os
from typing import Optional


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def select_device(prefer_gpu: bool = True) -> str:
    """Seleciona automaticamente GPU/CPU com base em disponibilidade."""
    override = os.getenv("DEVICE")
    if override:
        return override.strip().lower()
    if _env_bool("FORCE_CPU"):
        return "cpu"
    if not prefer_gpu:
        return "cpu"
    if find_spec("torch") is None:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    base_url: str
    chat_model: str
    embeddings_model: str


@dataclass(frozen=True)
class LocalEmbeddingsSettings:
    enabled: bool
    model_name: str
    provider: str
    api_url: Optional[str]


@dataclass(frozen=True)
class AppSettings:
    openai: OpenAISettings
    local_embeddings: LocalEmbeddingsSettings
    device: str


def load_settings() -> AppSettings:
    """Carrega configurações a partir de variáveis de ambiente."""
    openai = OpenAISettings(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL),
        embeddings_model=os.getenv(
            "OPENAI_EMBEDDINGS_MODEL", DEFAULT_OPENAI_EMBEDDINGS_MODEL
        ),
    )
    local_embeddings = LocalEmbeddingsSettings(
        enabled=_env_bool("LOCAL_EMBEDDINGS_ENABLED"),
        model_name=os.getenv("LOCAL_EMBEDDINGS_MODEL", DEFAULT_LOCAL_EMBEDDINGS_MODEL),
        provider=os.getenv("LOCAL_EMBEDDINGS_PROVIDER", "sentence-transformers"),
        api_url=os.getenv("LOCAL_EMBEDDINGS_API_URL"),
    )
    device = select_device(prefer_gpu=_env_bool("PREFER_GPU", True))
    return AppSettings(
        openai=openai,
        local_embeddings=local_embeddings,
        device=device,
    )
