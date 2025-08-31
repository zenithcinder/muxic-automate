"""
Text matching and scoring logic for YouTube Search to MP3 Downloader.
"""

import logging
import math
from typing import List, Dict, Any

from .config import get_runtime_config, get_embed_model, set_embed_model
from .utils import (
    _normalize_text,
    _strip_common_noise,
    _tokenize,
    _token_set_string,
    _non_alnum_ratio,
    _similarity,
    _token_jaccard,
    _build_query_variants,
)


def _duration_similarity_seconds(entry: dict, desired_seconds: int | None) -> float:
    """Calculate duration similarity score."""
    if not desired_seconds:
        return 0.0
    try:
        dur = entry.get("duration")
        if dur is None:
            return 0.0
        dur = int(dur)
        desired = max(30, int(desired_seconds))
        delta = abs(dur - desired)
        # 0 diff => 1.0, 60s diff => ~0.5, >= 5min diff => ~0.0
        return max(0.0, 1.0 - (delta / max(60.0, desired * 1.0)))
    except Exception:
        return 0.0


def _get_view_count(entry: dict) -> int:
    """Extract view count from entry."""
    try:
        vc = entry.get("view_count")
        return int(vc) if vc is not None else -1
    except Exception:
        return -1


def _views_score(entry: dict) -> float:
    """Scale views non-linearly to 0..1 using log10; handles missing as 0."""
    views = _get_view_count(entry)
    if views <= 0:
        return 0.0
    # log10 scaling: 1e3 -> ~0.5, 1e6 -> ~1.0 (cap at 1)
    return max(0.0, min(1.0, math.log10(views + 1) / 6.0))


def _get_preferred_device() -> str:
    """Determine the preferred device for the embed model, preferring GPU if available."""
    try:
        import torch

        if torch.cuda.is_available():
            # Check if CUDA is properly configured
            try:
                # Test if we can create a tensor on GPU
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor  # Clean up
                logging.info("GPU (CUDA) detected and available for AI matching")
                return "cuda"
            except Exception as gpu_error:
                logging.warning(
                    "CUDA available but failed to create test tensor: %s, falling back to CPU",
                    gpu_error
                )
                return "cpu"
        else:
            logging.info("No CUDA GPU available, using CPU for AI matching")
            return "cpu"
    except ImportError:
        # PyTorch not available, use CPU
        logging.warning("PyTorch not available, using CPU for AI matching")
        return "cpu"
    except Exception as e:
        # Any other error, fall back to CPU
        logging.warning("Error detecting GPU: %s, falling back to CPU", e)
        return "cpu"


def _ensure_embed_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load sentence-transformers model once if AI match is enabled.
    
    Handles PyTorch meta tensor issues with newer versions by using to_empty() method.
    """
    model = get_embed_model()
    if model is not None:
        return model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as import_error:  # noqa: BLE001
        logging.error(
            "AI matching requested but 'sentence-transformers' is not installed: %s",
            import_error,
        )
        return None

    # Try GPU first, fallback to CPU if needed
    preferred_device = _get_preferred_device()

    try:
        # Handle meta tensor issue with newer PyTorch versions
        try:
            if preferred_device == "cuda":
                logging.info("Attempting to load model on GPU (CUDA)...")
            model = SentenceTransformer(model_name, device=preferred_device)
        except RuntimeError as meta_error:
            if "meta tensor" in str(meta_error).lower():
                # Use to_empty() for meta tensor handling
                logging.info("Handling meta tensor issue with to_empty() method...")
                model = SentenceTransformer(model_name)
                model = model.to_empty(device=preferred_device)
            else:
                raise
        set_embed_model(model)
        logging.info(
            "AI matching: successfully loaded model on %s: %s", preferred_device, model_name
        )
    except Exception as load_error:  # noqa: BLE001
        logging.warning(
            "Failed to load AI model '%s' on %s: %s",
            model_name,
            preferred_device,
            load_error,
        )

        # If GPU failed, try CPU as fallback
        if preferred_device != "cpu":
            try:
                logging.info("Retrying with CPU device...")
                # Handle meta tensor issue with newer PyTorch versions
                try:
                    model = SentenceTransformer(model_name, device="cpu")
                except RuntimeError as meta_error:
                    if "meta tensor" in str(meta_error).lower():
                        # Use to_empty() for meta tensor handling
                        model = SentenceTransformer(model_name)
                        model = model.to_empty(device="cpu")
                    else:
                        raise
                set_embed_model(model)
                logging.info("AI matching: loaded model on cpu: %s", model_name)
            except Exception as cpu_load_error:  # noqa: BLE001
                logging.error(
                    "Failed to load AI model '%s' on CPU: %s",
                    model_name,
                    cpu_load_error,
                )
                # Continue to fallback model
        else:
            # CPU already failed, continue to fallback model
            pass

        # Fallback to a smaller, widely compatible model
        fallback = "sentence-transformers/all-MiniLM-L6-v2"
        if model_name != fallback:
            try:
                # Handle meta tensor issue with newer PyTorch versions
                try:
                    model = SentenceTransformer(fallback, device=preferred_device)
                except RuntimeError as meta_error:
                    if "meta tensor" in str(meta_error).lower():
                        # Use to_empty() for meta tensor handling
                        model = SentenceTransformer(fallback)
                        model = model.to_empty(device=preferred_device)
                    else:
                        raise
                set_embed_model(model)
                logging.warning(
                    "AI matching: falling back to model on %s: %s",
                    preferred_device,
                    fallback,
                )
            except Exception as fallback_error:  # noqa: BLE001
                # Try CPU for fallback model too
                if preferred_device != "cpu":
                    try:
                        # Handle meta tensor issue with newer PyTorch versions
                        try:
                            model = SentenceTransformer(fallback, device="cpu")
                        except RuntimeError as meta_error:
                            if "meta tensor" in str(meta_error).lower():
                                # Use to_empty() for meta tensor handling
                                model = SentenceTransformer(fallback)
                                model = model.to_empty(device="cpu")
                            else:
                                raise
                        set_embed_model(model)
                        logging.warning(
                            "AI matching: falling back to model on cpu: %s", fallback
                        )
                    except Exception as cpu_fallback_error:  # noqa: BLE001
                        logging.error(
                            "Failed to load fallback AI model '%s' on CPU: %s",
                            fallback,
                            cpu_fallback_error,
                        )
                        set_embed_model(None)
                else:
                    logging.error(
                        "Failed to load fallback AI model '%s': %s",
                        fallback,
                        fallback_error,
                    )
                    set_embed_model(None)
        else:
            set_embed_model(None)
    return get_embed_model()


def _ai_similarity_score(query_text: str, title: str, authors_text: str) -> float:
    """Semantic similarity using sentence-transformers; returns 0..1 (approx).

    Falls back to 0 if model is unavailable.
    """
    config = get_runtime_config()
    if not config.use_ai_match:
        return 0.0
    model = _ensure_embed_model(
        config.ai_model_name or "sentence-transformers/all-MiniLM-L6-v2"
    )
    if model is None:
        return 0.0
    try:
        # Concatenate title and authors to better capture semantics
        import numpy as np  # type: ignore

        query_vec = model.encode([query_text], normalize_embeddings=True)
        cand_vec = model.encode(
            [f"{title} | {authors_text}"], normalize_embeddings=True
        )
        # Cosine similarity because vectors are normalized
        sim = float(np.dot(query_vec[0], cand_vec[0]))
        # Clamp numerical edge cases
        return max(0.0, min(1.0, sim))
    except Exception as error:  # noqa: BLE001
        logging.debug("AI similarity failed, ignoring: %s", error)
        return 0.0


def _extract_authors_text(entry: dict) -> str:
    """Collect plausible author/artist/channel fields into one string for matching."""
    parts: list[str] = []
    artist = entry.get("artist")
    if isinstance(artist, str):
        parts.append(artist)
    artists = entry.get("artists")
    if isinstance(artists, list):
        parts.extend([a for a in artists if isinstance(a, str)])
    album_artist = entry.get("album_artist")
    if isinstance(album_artist, str):
        parts.append(album_artist)
    channel = entry.get("channel")
    if isinstance(channel, str):
        parts.append(channel)
    uploader = entry.get("uploader")
    if isinstance(uploader, str):
        parts.append(uploader)
    return " ".join(parts)


def _extract_author_tokens_from_query(query_text: str) -> list[str]:
    """Heuristically extract author tokens from a query that may contain title and author.

    Looks for separators like ' - ', ' by ', parentheses, and common joiners like 'feat/ft'.
    """
    text = _normalize_text(query_text)
    # Split by common separators hinting at author vs title
    candidates: list[str] = []
    if " by " in text:
        # title by author
        parts = text.split(" by ", 1)
        candidates.append(parts[1])
    if " - " in text:
        parts = text.split(" - ", 1)
        # ambiguous; keep right side as probable author if short-ish
        right = parts[1]
        if len(right) <= 60:
            candidates.append(right)
    # Inside parentheses at the end often has extra info; ignore
    # Tokenize
    tokens: list[str] = []
    for chunk in candidates:
        tokens.extend(_tokenize(chunk))
    # Remove common non-author tokens
    noisy = {
        "official",
        "video",
        "lyrics",
        "visualizer",
        "audio",
        "mv",
        "hd",
        "remix",
        "cover",
        "live",
    }
    filtered = [t for t in tokens if t not in noisy]
    return filtered


def _extract_author_tokens_from_entry(entry: dict) -> list[str]:
    """Extract author tokens from entry."""
    authors_full = _extract_authors_text(entry)
    return _tokenize(authors_full)


def _is_short_video(entry: dict) -> bool:
    """Heuristic to detect YouTube Shorts.

    - URLs containing '/shorts/'
    - Or very short duration (<= 65s) when available
    - Or extractor key hints
    """
    try:
        url = entry.get("webpage_url") or entry.get("url") or ""
        if isinstance(url, str) and "/shorts/" in url:
            return True
    except Exception:
        pass

    try:
        duration = entry.get("duration")
        if duration is not None and int(duration) <= 65:
            return True
    except Exception:
        pass

    try:
        extractor_key = entry.get("extractor_key") or entry.get("extractor")
        if isinstance(extractor_key, str) and "short" in extractor_key.lower():
            return True
    except Exception:
        pass

    return False


def _is_long_video(entry: dict, max_seconds: int = 600) -> bool:
    """Detect long-form videos (default > 10 minutes)."""
    try:
        duration = entry.get("duration")
        if duration is not None and int(duration) > int(max_seconds):
            return True
    except Exception:
        pass
    return False


def _title_flags(text: str) -> dict:
    """Extract flags from title text."""
    norm = _normalize_text(text)
    return {
        "official": "official" in norm,
        "live": "live" in norm,
        "full_album": "full album" in norm or "album" in norm,
        "mix": "mix" in norm or "playlist" in norm,
        "concert": "concert" in norm,
        "cover": "cover" in norm,
        "remix": "remix" in norm,
        "visualizer": "visualizer" in norm,
        "lyrics": "lyric" in norm or "lyrics" in norm,
    }


def _should_skip_entry(entry: dict, query_text: str | None = None) -> bool:
    """Return True when an entry should be filtered out from candidates.

    Rules:
    - Exclude Shorts
    - Exclude long-form (>10 minutes)
    - Exclude obvious non-single content: full album, mix/playlist, concert, visualizer
    - Exclude live, cover, remix by default
    """
    if _is_short_video(entry):
        return True
    if _is_long_video(entry):
        return True

    title = (entry.get("title") or "").strip()
    flags = _title_flags(title)

    # If the user's query explicitly asks for these, do not skip for that reason
    query_norm = _normalize_text(query_text) if query_text else ""

    def query_mentions(term: str) -> bool:
        return bool(term) and (term in query_norm)

    if flags["full_album"] and not query_mentions("album"):
        return True
    if flags["mix"] and not (query_mentions("mix") or query_mentions("playlist")):
        return True
    if flags["concert"] and not query_mentions("concert"):
        return True
    if flags["visualizer"] and not query_mentions("visualizer"):
        return True
    if flags["live"] and not query_mentions("live"):
        return True
    if flags["cover"] and not query_mentions("cover"):
        return True
    if flags["remix"] and not query_mentions("remix"):
        return True

    return False


def _find_exact_matches(entries: list, query_text: str) -> list[dict]:
    """Return entries whose titles match the query strongly (variant and token-set exact)."""
    query_variants = _build_query_variants(query_text)
    exact = []
    for e in entries:
        title = e.get("title") or ""
        title_norm = _normalize_text(title)
        title_token_set = _token_set_string(_tokenize(title))
        if title_norm in query_variants or title_token_set in query_variants:
            exact.append(e)
    return exact


def pick_entry(entries: list, strategy: str) -> dict | None:
    """Pick one entry from a ytsearch result according to strategy.

    Strategies:
    - 'first': first entry returned by search
    - 'best': pick exact title match if present; otherwise closest title to query
    """
    if not entries:
        return None
    if strategy == "first":
        return entries[0]

    if strategy == "best":
        # Try to get the original query from the entries if available via _query hint
        # When not available, we will approximate using the most common title tokens.
        # Instead, we will pass the query via a closure from search_video_url.
        # To enable that, we expect the caller to have attached '__query__' to each entry.
        # If it's not present, we fall back to using the first entry title as an anchor,
        # which still gives a reasonable similarity ordering.
        query_text = None
        for e in entries:
            if "__query__" in e:
                query_text = e["__query__"]
                break

        # If not provided, approximate with first title
        if query_text is None:
            query_text = entries[0].get("title") or ""

        query_norm = _normalize_text(query_text)
        query_variants = _build_query_variants(query_text)

        # Prefer exact normalized title match (across variants); if many exact, prefer those whose authors match query
        exact_matches = []
        for e in entries:
            title = e.get("title") or ""
            title_norm = _normalize_text(title)
            title_token_set = _token_set_string(_tokenize(title))
            if title_norm in query_variants or title_token_set in query_variants:
                exact_matches.append(e)

        if exact_matches:
            # Rank exact matches by author overlap with the query (if any)
            def exact_rank(e: dict) -> float:
                authors_text = _extract_authors_text(e)
                if not authors_text:
                    return 0.0
                return _similarity(_normalize_text(authors_text), query_norm)

            return sorted(exact_matches, key=exact_rank, reverse=True)[0]

        # No exact match: compute similarity to title and channel/uploader
        def score(e: dict) -> float:
            title = e.get("title") or ""
            authors_full = _extract_authors_text(e)
            entry_author_tokens = _extract_author_tokens_from_entry(e)
            query_author_tokens = _extract_author_tokens_from_query(query_text)
            # Multiple similarity signals
            title_norm = _normalize_text(title)
            title_noise_stripped = _strip_common_noise(title)
            title_tokens = _tokenize(title)
            token_sort_title = " ".join(sorted(title_tokens))
            token_set_title = _token_set_string(title_tokens)
            query_tokens = _tokenize(query_text)

            # Base: normalized similarity
            s_base = _similarity(title_norm, query_norm)
            # Noise stripped often helps
            s_noise = _similarity(title_noise_stripped, _strip_common_noise(query_text))
            # Token sort/set similarities approximate fuzzywuzzy behavior
            s_token_sort = _similarity(
                token_sort_title, " ".join(sorted(_tokenize(query_text)))
            )
            s_token_set = _similarity(
                token_set_title, _token_set_string(_tokenize(query_text))
            )
            # Token overlap (Jaccard)
            s_jaccard = _token_jaccard(title_tokens, query_tokens)
            # Substring bonus when query is contained in title (or vice versa)
            contains_bonus = (
                0.1 if (query_norm in title_norm or title_norm in query_norm) else 0.0
            )
            # Author relevance (channel/uploader/artist)
            s_author = _similarity(_normalize_text(authors_full), query_norm) * 0.2
            # Author token overlap bonus
            s_author_tok = _token_jaccard(entry_author_tokens, query_author_tokens)
            # AI semantic similarity (optional)
            s_ai = _ai_similarity_score(query_text, title, authors_full) * 0.6
            # Length similarity (favor titles close to query length)
            len_title = max(1, len(title_norm))
            len_query = max(1, len(query_norm))
            s_len = 1.0 - (abs(len_title - len_query) / max(len_title, len_query))
            # Non-alphanumeric penalty for noisy/gibberish titles
            noise_penalty = -0.15 * _non_alnum_ratio(title)
            # Prefix/suffix bonus when title starts/ends closely with query
            prefix_bonus = 0.07 if title_norm.startswith(query_norm) else 0.0
            suffix_bonus = 0.04 if title_norm.endswith(query_norm) else 0.0

            # Weighted sum; keep within a reasonable range. Preference for official music.
            # Include popularity via views and duration alignment to Spotify when available
            s_views = _views_score(e)
            spotify_info = e.get("__spotify__") or {}
            desired_sec = (
                spotify_info.get("duration_sec")
                if isinstance(spotify_info, dict)
                else None
            )
            s_dur = _duration_similarity_seconds(e, desired_sec)
            flags = _title_flags(title)
            official_bonus = 0.08 if flags.get("official") else 0.0
            live_penalty = -0.06 if flags.get("live") else 0.0
            mix_penalty = -0.06 if flags.get("mix") else 0.0
            album_penalty = -0.05 if flags.get("full_album") else 0.0
            visualizer_penalty = -0.03 if flags.get("visualizer") else 0.0

            base_score = (
                0.28 * s_base
                + 0.16 * s_noise
                + 0.16 * s_token_sort
                + 0.16 * s_token_set
                + 0.12 * s_jaccard
                + 0.08 * s_len
                + contains_bonus
                + s_author
                + 0.15 * s_author_tok
                + prefix_bonus
                + suffix_bonus
                + noise_penalty
                + s_ai
                + 0.10 * s_views
                + 0.10 * s_dur
            )
            return (
                base_score
                + official_bonus
                + live_penalty
                + mix_penalty
                + album_penalty
                + visualizer_penalty
            )

        # If debugging is enabled, log top-K with score breakdown
        config = get_runtime_config()
        if config.debug_matching:
            annotated = []
            for e in entries:
                title = e.get("title") or ""
                authors_full = _extract_authors_text(e)
                entry_author_tokens = _extract_author_tokens_from_entry(e)
                query_author_tokens = _extract_author_tokens_from_query(query_text)
                title_norm = _normalize_text(title)
                title_noise_stripped = _strip_common_noise(title)
                title_tokens = _tokenize(title)
                token_sort_title = " ".join(sorted(title_tokens))
                token_set_title = _token_set_string(title_tokens)
                query_tokens = _tokenize(query_text)
                s_base = _similarity(title_norm, query_norm)
                s_noise = _similarity(
                    title_noise_stripped, _strip_common_noise(query_text)
                )
                s_token_sort = _similarity(
                    token_sort_title, " ".join(sorted(_tokenize(query_text)))
                )
                s_token_set = _similarity(
                    token_set_title, _token_set_string(_tokenize(query_text))
                )
                s_jaccard = _token_jaccard(title_tokens, query_tokens)
                contains_bonus = (
                    0.1
                    if (query_norm in title_norm or title_norm in query_norm)
                    else 0.0
                )
                s_author = _similarity(_normalize_text(authors_full), query_norm) * 0.2
                s_author_tok = _token_jaccard(entry_author_tokens, query_author_tokens)
                s_ai = _ai_similarity_score(query_text, title, authors_full) * 0.6
                len_title = max(1, len(title_norm))
                len_query = max(1, len(query_norm))
                s_len = 1.0 - (abs(len_title - len_query) / max(len_title, len_query))
                noise_penalty = -0.15 * _non_alnum_ratio(title)
                prefix_bonus = 0.7 if title_norm.startswith(query_norm) else 0.0
                suffix_bonus = 0.4 if title_norm.endswith(query_norm) else 0.0
                flags = _title_flags(title)
                official_bonus = 0.08 if flags.get("official") else 0.0
                live_penalty = -0.06 if flags.get("live") else 0.0
                mix_penalty = -0.06 if flags.get("mix") else 0.0
                album_penalty = -0.05 if flags.get("full_album") else 0.0
                visualizer_penalty = -0.03 if flags.get("visualizer") else 0.0
                base_score = (
                    0.28 * s_base
                    + 0.16 * s_noise
                    + 0.16 * s_token_sort
                    + 0.16 * s_token_set
                    + 0.12 * s_jaccard
                    + 0.08 * s_len
                    + contains_bonus
                    + s_author
                    + 0.15 * s_author_tok
                    + (0.0 if s_ai == 0.0 else s_ai)
                    + (0.7 if title_norm.startswith(query_norm) else 0.0)
                    + (0.4 if title_norm.endswith(query_norm) else 0.0)
                    + noise_penalty
                )
                s_views = _views_score(e)
                total = (
                    base_score
                    + official_bonus
                    + live_penalty
                    + mix_penalty
                    + album_penalty
                    + visualizer_penalty
                    + 0.10 * s_views
                )
                annotated.append(
                    (
                        total,
                        title,
                        {
                            "s_base": s_base,
                            "s_noise": s_noise,
                            "s_token_sort": s_token_sort,
                            "s_token_set": s_token_set,
                            "s_jaccard": s_jaccard,
                            "s_len": s_len,
                            "contains_bonus": contains_bonus,
                            "s_author": s_author,
                            "s_author_tok": s_author_tok,
                            "s_ai": s_ai,
                            "s_views": s_views,
                            "official_bonus": official_bonus,
                            "live_penalty": live_penalty,
                            "mix_penalty": mix_penalty,
                            "album_penalty": album_penalty,
                            "visualizer_penalty": visualizer_penalty,
                            "noise_penalty": noise_penalty,
                        },
                    )
                )
            annotated.sort(key=lambda x: x[0], reverse=True)
            top_k = config.debug_top_k
            logging.info("Matching breakdown for query: %s", query_text)
            for rank, (total, title, parts) in enumerate(annotated[:top_k], start=1):
                logging.info("%d) score=%.3f | %s", rank, total, title)
                logging.debug("  details: %s", parts)

        return sorted(entries, key=score, reverse=True)[0]

    # Default fallback
    return entries[0]


def check_gpu_availability() -> dict:
    """Check GPU availability and provide detailed diagnostics."""
    diagnostics = {
        "gpu_available": False,
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory": None,
        "error": None
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            diagnostics["cuda_available"] = True
            
            try:
                # Test GPU functionality
                test_tensor = torch.tensor([1.0], device="cuda")
                del test_tensor
                
                diagnostics["gpu_available"] = True
                diagnostics["device"] = "cuda"
                
                # Get GPU info
                if torch.cuda.device_count() > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    diagnostics["gpu_name"] = gpu_name
                    
                    # Get GPU memory info
                    try:
                        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                        diagnostics["gpu_memory"] = {
                            "allocated_gb": round(memory_allocated, 2),
                            "reserved_gb": round(memory_reserved, 2)
                        }
                    except Exception:
                        pass
                        
            except Exception as e:
                diagnostics["error"] = str(e)
                diagnostics["gpu_available"] = False
                diagnostics["device"] = "cpu"
                
    except ImportError:
        diagnostics["error"] = "PyTorch not available"
    except Exception as e:
        diagnostics["error"] = str(e)
    
    return diagnostics


__all__ = [
    "_duration_similarity_seconds",
    "_get_view_count",
    "_views_score",
    "_get_preferred_device",
    "_ensure_embed_model",
    "_ai_similarity_score",
    "_extract_authors_text",
    "_extract_author_tokens_from_query",
    "_extract_author_tokens_from_entry",
    "_is_short_video",
    "_is_long_video",
    "_title_flags",
    "_should_skip_entry",
    "_find_exact_matches",
    "pick_entry",
    "check_gpu_availability",
]
