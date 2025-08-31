"""
Utility functions for YouTube Search to MP3 Downloader.
"""

import logging
import sys
import unicodedata
import re
import csv
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def configure_logging(verbosity: str, log_file: Path, enable_csv_logging: bool = False) -> None:
    """Configure root logger with console and file handlers.

    This function sets up a comprehensive logging system that writes to both
    the console and a log file. It configures the root logger to handle
    different verbosity levels and formats messages consistently.

    Parameters
    ----------
    verbosity: str
        One of DEBUG, INFO, WARNING, ERROR.
    log_file: Path
        File path to write detailed logs.
    enable_csv_logging: bool
        Whether to enable CSV logging for concurrent operations.

    Examples
    --------
    >>> configure_logging("INFO", Path("logs/app.log"))
    # Sets up logging with INFO level, writes to console and logs/app.log
    
    >>> configure_logging("DEBUG", Path("debug.log"))
    # Sets up detailed DEBUG logging for troubleshooting
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
    
    # Set CSV logging flag in global config if enabled
    if enable_csv_logging:
        root_logger.csv_logging_enabled = True
        root_logger.csv_log_file = log_file.parent / f"{log_file.stem}_concurrent.csv"
        # Initialize CSV file with headers
        _init_csv_log_file(root_logger.csv_log_file)


def _init_csv_log_file(csv_file: Path) -> None:
    """Initialize CSV log file with headers."""
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Only write headers if file doesn't exist or is empty
    if not csv_file.exists() or csv_file.stat().st_size == 0:
        with csv_file.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'thread_id',
                'query_index',
                'query',
                'url',
                'success',
                'reason',
                'duration_ms',
                'status'
            ])


def log_download_result_csv(
    query: str,
    url: Optional[str],
    success: bool,
    reason: Optional[str] = None,
    query_index: Optional[int] = None,
    duration_ms: Optional[float] = None,
    status: str = "completed"
) -> None:
    """Log download result in CSV format for concurrent operations.
    
    This function logs download results in a structured CSV format that makes it
    easy to analyze concurrent operations, identify failures, and sort by various criteria.
    
    Parameters
    ----------
    query: str
        The search query that was processed.
    url: Optional[str]
        The YouTube URL that was downloaded (or None if failed).
    success: bool
        Whether the download was successful.
    reason: Optional[str]
        Error reason if download failed.
    query_index: Optional[int]
        The index/position of this query in the batch.
    duration_ms: Optional[float]
        How long the operation took in milliseconds.
    status: str
        Status of the operation (e.g., "started", "completed", "failed").
    """
    logger = logging.getLogger()
    
    # Check if CSV logging is enabled
    if not hasattr(logger, 'csv_logging_enabled') or not logger.csv_logging_enabled:
        return
    
    try:
        csv_file = logger.csv_log_file
        thread_id = threading.get_ident()
        timestamp = datetime.now().isoformat()
        
        with csv_file.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                thread_id,
                query_index or '',
                query,
                url or '',
                '1' if success else '0',  # Use 1/0 for boolean in CSV
                reason or '',
                f"{duration_ms:.2f}" if duration_ms is not None else '',
                status
            ])
    except Exception as e:
        # Fallback to regular logging if CSV logging fails
        logging.warning("Failed to write CSV log entry: %s", e)


def get_csv_log_file() -> Optional[Path]:
    """Get the path to the CSV log file if CSV logging is enabled."""
    logger = logging.getLogger()
    if hasattr(logger, 'csv_logging_enabled') and logger.csv_logging_enabled:
        return logger.csv_log_file
    return None


def read_queries(file_path: Path) -> List[str]:
    """Read non-empty, non-comment lines from a file as queries.
    
    This function reads a text file and extracts valid search queries,
    skipping empty lines and comments (lines starting with #).
    Useful for batch processing multiple search terms.

    Parameters
    ----------
    file_path: Path
        Path to the text file containing search queries.

    Returns
    -------
    List[str]
        List of non-empty, non-comment lines from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist.

    Examples
    --------
    # File: queries.txt
    # # This is a comment
    # 
    # Ed Sheeran Shape of You
    # The Weeknd Blinding Lights
    # 
    # Another comment
    
    >>> queries = read_queries(Path("queries.txt"))
    >>> print(queries)
    ['Ed Sheeran Shape of You', 'The Weeknd Blinding Lights']
    """
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
    """Best-effort check that ffmpeg is available on PATH.
    
    This function checks if ffmpeg is installed and accessible from the
    system PATH. It logs a warning if ffmpeg is not found, as it's
    required for audio extraction and conversion operations.

    Examples
    --------
    >>> ensure_ffmpeg_available()
    # If ffmpeg is found: No output
    # If ffmpeg is missing: WARNING: ffmpeg not found on PATH...
    """
    from shutil import which

    if which("ffmpeg") is None:
        logging.warning(
            "ffmpeg not found on PATH. Audio extraction will fail. Please install ffmpeg."
        )


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for robust matching.
    
    This function normalizes text for consistent comparison by:
    1. Converting to lowercase using Unicode-aware case folding
    2. Normalizing Unicode characters (NFKC form)
    3. Removing all non-alphanumeric characters
    4. Collapsing multiple spaces into single spaces
    
    This ensures that "Hello, World!" and "hello world" are treated as equivalent.

    Parameters
    ----------
    text: str
        The text string to normalize.

    Returns
    -------
    str
        Normalized text string.

    Examples
    --------
    >>> _normalize_text("Hello, World! (Official Video)")
    'hello world official video'
    
    >>> _normalize_text("The Beatles - Hey Jude (Remix)")
    'the beatles hey jude remix'
    
    >>> _normalize_text("   Multiple    Spaces   ")
    'multiple spaces'
    """
    # Unicode-aware folding and normalization for non-Latin scripts (e.g., Cyrillic)
    lowered = unicodedata.normalize("NFKC", text).casefold().strip()
    # Replace any non-alphanumeric (unicode aware) with space
    cleaned = re.sub(r"[^\w]+", " ", lowered, flags=re.UNICODE)
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", cleaned).strip()


def _strip_common_noise(text: str) -> str:
    """Remove frequent noise terms in YouTube titles (e.g., official video, lyrics).
    
    This function removes common terms that appear in YouTube video titles
    but don't contribute to the actual song/artist identification. It helps
    improve matching accuracy by focusing on the core content.

    Parameters
    ----------
    text: str
        The text string to clean.

    Returns
    -------
    str
        Text with noise terms removed and normalized.

    Examples
    --------
    >>> _strip_common_noise("Shape of You - Ed Sheeran (Official Video)")
    'shape of you ed sheeran'
    
    >>> _strip_common_noise("Blinding Lights - The Weeknd (Lyrics)")
    'blinding lights the weeknd'
    
    >>> _strip_common_noise("Song Title (Official Music Video) [HD]")
    'song title'
    """
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
    """Tokenize text into words.
    
    This function splits normalized text into individual word tokens,
    filtering out empty strings. Used for more granular text analysis
    and comparison.

    Parameters
    ----------
    text: str
        The text string to tokenize.

    Returns
    -------
    list[str]
        List of word tokens.

    Examples
    --------
    >>> _tokenize("Hello, World! How are you?")
    ['hello', 'world', 'how', 'are', 'you']
    
    >>> _tokenize("The Beatles - Hey Jude")
    ['the', 'beatles', 'hey', 'jude']
    """
    norm = _normalize_text(text)
    return [t for t in norm.split(" ") if t]


def _token_set_string(tokens: list[str]) -> str:
    """Convert token list to sorted unique string.
    
    This function creates a canonical representation of tokens by:
    1. Removing duplicates (converting to set)
    2. Sorting alphabetically
    3. Joining with spaces
    
    This ensures that "hello world" and "world hello" produce the same result.

    Parameters
    ----------
    tokens: list[str]
        List of word tokens.

    Returns
    -------
    str
        Sorted, unique tokens joined with spaces.

    Examples
    --------
    >>> _token_set_string(['hello', 'world', 'hello'])
    'hello world'
    
    >>> _token_set_string(['the', 'beatles', 'hey', 'jude'])
    'beatles hey jude the'
    """
    return " ".join(sorted(set(tokens)))


def _non_alnum_ratio(text: str) -> float:
    """Calculate ratio of non-alphanumeric characters.
    
    This function calculates what percentage of characters in a string
    are non-alphanumeric (punctuation, symbols, etc.). Useful for
    detecting overly complex or noisy text that might be less reliable
    for matching.

    Parameters
    ----------
    text: str
        The text string to analyze.

    Returns
    -------
    float
        Ratio of non-alphanumeric characters (0.0 to 1.0).

    Examples
    --------
    >>> _non_alnum_ratio("Hello World")
    0.0  # No non-alphanumeric characters
    
    >>> _non_alnum_ratio("Hello, World!")
    0.25  # 3 non-alphanumeric chars out of 12 total
    
    >>> _non_alnum_ratio("!!!@@@###")
    1.0  # All characters are non-alphanumeric
    """
    if not text:
        return 0.0
    total = len(text)
    non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return non_alnum / max(1, total)


def _similarity(a: str, b: str) -> float:
    """Return similarity ratio in [0, 1] using difflib.
    
    This function calculates the similarity between two strings using
    Python's difflib.SequenceMatcher. Returns a value between 0.0
    (completely different) and 1.0 (identical).

    Parameters
    ----------
    a: str
        First string to compare.
    b: str
        Second string to compare.

    Returns
    -------
    float
        Similarity ratio between 0.0 and 1.0.

    Examples
    --------
    >>> _similarity("hello world", "hello world")
    1.0  # Identical strings
    
    >>> _similarity("hello world", "hello there")
    0.5454545454545454  # Partial similarity
    
    >>> _similarity("hello", "goodbye")
    0.0  # No similarity
    """
    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_jaccard(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Calculate Jaccard similarity between token sets.
    
    This function calculates the Jaccard similarity coefficient between
    two sets of tokens. The Jaccard index is the size of the intersection
    divided by the size of the union of the two sets.
    
    Jaccard = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    a_tokens: list[str]
        First list of tokens.
    b_tokens: list[str]
        Second list of tokens.

    Returns
    -------
    float
        Jaccard similarity between 0.0 and 1.0.

    Examples
    --------
    >>> _token_jaccard(['hello', 'world'], ['hello', 'world'])
    1.0  # Identical token sets
    
    >>> _token_jaccard(['hello', 'world'], ['hello', 'there'])
    0.3333333333333333  # 1 common token out of 3 total unique tokens
    
    >>> _token_jaccard(['hello'], ['world'])
    0.0  # No common tokens
    """
    set_a = set(a_tokens)
    set_b = set(b_tokens)
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(1, union)


def _build_query_variants(query_text: str) -> list[str]:
    """Build normalized variants of the query for robust exact matching.
    
    This function creates multiple normalized versions of a query to
    improve matching accuracy. It generates variants that handle
    different ways the same content might be formatted or described.

    Parameters
    ----------
    query_text: str
        The original query text.

    Returns
    -------
    list[str]
        List of normalized query variants.

    Examples
    --------
    >>> _build_query_variants("Shape of You - Ed Sheeran (Official Video)")
    ['shape of you ed sheeran official video', 'shape of you ed sheeran', 'ed sheeran shape of you']
    
    >>> _build_query_variants("The Beatles - Hey Jude")
    ['the beatles hey jude', 'hey jude', 'beatles hey jude the']
    """
    variants = set()
    variants.add(_normalize_text(query_text))
    variants.add(_strip_common_noise(query_text))
    # Token set (ignoring duplicates and order)
    variants.add(_token_set_string(_tokenize(query_text)))
    return [v for v in variants if v]


def _extract_title_from_query(query_text: str) -> str:
    """Extract just the title part from a query, removing author information.
    
    This function attempts to extract the song title from a query that
    might contain artist information. It removes common author separators
    and artist indicators to isolate the actual title.

    Parameters
    ----------
    query_text: str
        The query text that may contain both title and artist information.

    Returns
    -------
    str
        The extracted title portion of the query.

    Examples
    --------
    >>> _extract_title_from_query("Shape of You by Ed Sheeran")
    'shape of you'
    
    >>> _extract_title_from_query("Blinding Lights - The Weeknd")
    'blinding lights'
    
    >>> _extract_title_from_query("Hey Jude (The Beatles)")
    'hey jude'
    
    >>> _extract_title_from_query("Song Title feat. Artist")
    'song title'
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
    "log_download_result_csv",
    "get_csv_log_file",
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
