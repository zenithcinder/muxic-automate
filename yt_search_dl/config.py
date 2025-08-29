"""
Configuration and data models for YouTube Search to MP3 Downloader.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DownloadResult:
    """Result of a download operation."""
    query: str
    url: str | None
    success: bool
    reason: str | None = None


@dataclass
class RuntimeConfig:
    """Runtime configuration for the application."""
    search_count: int = 20
    select_strategy: str = "best"
    use_ai_match: bool = False
    ai_model_name: str | None = None
    debug_matching: bool = False
    debug_top_k: int = 10
    use_spotify: bool = False
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None
    spotify_limit: int = 5
    spotify_min_score: float = 0.35
    deep_search: bool = False
    search_without_authors: bool = False


# Module-global runtime configuration, populated in main()
_RUNTIME_CONFIG: RuntimeConfig = RuntimeConfig()
_EMBED_MODEL: Any = None  # Lazy-initialized sentence-transformers model when AI match is enabled
_SPOTIFY_TOKEN: dict | None = None  # {"access_token": str, "expires_at": int}


def get_runtime_config() -> RuntimeConfig:
    """Get the current runtime configuration."""
    return _RUNTIME_CONFIG


def set_runtime_config(config: RuntimeConfig) -> None:
    """Set the runtime configuration."""
    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = config


def get_embed_model():
    """Get the current embedding model."""
    return _EMBED_MODEL


def set_embed_model(model: Any) -> None:
    """Set the embedding model."""
    global _EMBED_MODEL
    _EMBED_MODEL = model


def get_spotify_token() -> dict | None:
    """Get the current Spotify token."""
    return _SPOTIFY_TOKEN


def set_spotify_token(token: dict | None) -> None:
    """Set the Spotify token."""
    global _SPOTIFY_TOKEN
    _SPOTIFY_TOKEN = token
