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
    # Audio quality settings
    audio_quality: str = "192"  # kbps or "best"
    audio_format: str = "mp3"   # "mp3", "m4a", "opus", "flac"
    # Authentication settings
    cookies_file: str | None = None  # Path to cookies.txt file
    # Google search settings
    use_google_search: bool = False
    google_api_key: str | None = None
    google_search_engine_id: str | None = None
    google_min_confidence: float = 0.3
    use_google_search_fallback: bool = False
    filter_queries_with_google: bool = False
    use_llm_google_parsing: bool = False
    llm_api_key: str | None = None
    llm_model: str = "gpt-3.5-turbo"
    llm_base_url: str | None = None  # For local LLMs like Ollama


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
