"""
Utility functions for YouTube Search to MP3 Downloader.
"""

import logging
import sys
import unicodedata
import re
from pathlib import Path
from typing import List


def configure_logging(verbosity: str, log_file: Path) -> None:
    """Configure root logger with console and file handlers.

    Parameters
    ----------
    verbosity: str
        One of DEBUG, INFO, WARNING, ERROR.
    log_file: Path
        File path to write detailed logs.
    """
    level = getattr(logging, verbosity.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if reconfiguring
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def read_queries(file_path: Path) -> List[str]:
    """Read non-empty, non-comment lines from a file as queries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    queries: List[str] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(line)
    return queries


def ensure_ffmpeg_available() -> None:
    """Best-effort check that ffmpeg is available on PATH."""
    from shutil import which

    if which("ffmpeg") is None:
        logging.warning(
            "ffmpeg not found on PATH. Audio extraction will fail. Please install ffmpeg."
        )


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for robust matching."""
    # Unicode-aware folding and normalization for non-Latin scripts (e.g., Cyrillic)
    lowered = unicodedata.normalize("NFKC", text).casefold().strip()
    # Replace any non-alphanumeric (unicode aware) with space
    cleaned = re.sub(r"[^\w]+", " ", lowered, flags=re.UNICODE)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_common_noise(text: str) -> str:
    """Remove frequent noise terms in YouTube titles (e.g., official video, lyrics)."""
    noise_patterns = [
        r"\bofficial\b",
        r"\bvideo\b",
        r"\bofficial\s*video\b",
        r"\blyrics?\b",
        r"\bvisualizer\b",
        r"\baudio\b",
        r"\bmv\b",
        r"\bhd\b",
        r"\b4k\b",
        r"\bremix\b",
        r"\bcover\b",
        r"\blive\b",
        r"\b(feat|ft)\.?\b",
        r"\bofficial\s*music\s*video\b",
        r"\bclip\b",
    ]
    without_brackets = re.sub(r"[\(\)\[\]\{\}]+", " ", text)
    result = without_brackets
    for pat in noise_patterns:
        result = re.sub(pat, " ", result, flags=re.IGNORECASE)
    return _normalize_text(result)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    norm = _normalize_text(text)
    return [t for t in norm.split(" ") if t]


def _token_set_string(tokens: list[str]) -> str:
    """Convert token list to sorted unique string."""
    return " ".join(sorted(set(tokens)))


def _non_alnum_ratio(text: str) -> float:
    """Calculate ratio of non-alphanumeric characters."""
    if not text:
        return 0.0
    total = len(text)
    non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return non_alnum / max(1, total)


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio in [0, 1] using difflib."""
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Calculate Jaccard similarity between token sets."""
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(1, union)


def _build_query_variants(query_text: str) -> list[str]:
    """Build normalized variants of the query for robust exact matching."""
    variants = set()
    variants.add(_normalize_text(query_text))
    variants.add(_strip_common_noise(query_text))
    # Token set (ignoring duplicates and order)
    variants.add(_token_set_string(_tokenize(query_text)))
    return [v for v in variants if v]


def _extract_title_from_query(query_text: str) -> str:
    """Extract just the title part from a query, removing author information.
    
    This function removes common author separators and returns just the title portion.
    """
    text = _normalize_text(query_text)
    
    # Remove author parts based on common separators
    if " by " in text:
        # "title by author" -> "title"
        parts = text.split(" by ", 1)
        return parts[0].strip()
    
    if " - " in text:
        # "title - author" -> "title" (assuming title is on the left)
        parts = text.split(" - ", 1)
        return parts[0].strip()
    
    # Remove text in parentheses at the end (often contains artist info)
    if "(" in text and text.rfind("(") > len(text) // 2:
        # Only remove if parentheses are in the latter half of the text
        text = text[:text.rfind("(")].strip()
    
    # Remove common artist indicators
    artist_indicators = ["feat.", "ft.", "featuring", "with", "&", "feat", "ft"]
    for indicator in artist_indicators:
        if f" {indicator} " in text:
            parts = text.split(f" {indicator} ", 1)
            text = parts[0].strip()
            break
    
    return text


# Export commonly used functions
__all__ = [
    "configure_logging",
    "read_queries", 
    "ensure_ffmpeg_available",
    "_normalize_text",
    "_strip_common_noise",
    "_tokenize",
    "_token_set_string",
    "_non_alnum_ratio",
    "_similarity",
    "_token_jaccard",
    "_build_query_variants",
    "_extract_title_from_query",
]
