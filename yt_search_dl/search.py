"""
YouTube search functionality for YouTube Search to MP3 Downloader.
"""

import logging
import sys
from typing import List

# yt-dlp will be imported when needed in functions

from .config import get_runtime_config
from .matching import pick_entry, _should_skip_entry
from .spotify import _spotify_search_first_track
from .utils import _extract_title_from_query


def search_video_url(query: str, search_count: int, strategy: str) -> str | None:
    """Search multiple YouTube results and select one URL based on strategy."""
    # Bound search_count to reasonable limits for reliability
    n = max(1, min(int(search_count), 50))
    
    # Apply deep search if enabled
    config = get_runtime_config()
    if config.deep_search:
        n = min(n * 2, 50)  # Double the search count, but cap at 50
        logging.debug("Deep search enabled: using %d results", n)
    
    # Optionally enrich query via Spotify
    enriched = _spotify_search_first_track(query)
    if enriched:
        title = enriched.get("title") or ""
        artists = " ".join(enriched.get("artists") or [])
        query = f"{title} {artists}".strip()
    
    # Apply title-only search if enabled
    if config.search_without_authors:
        original_query = query
        title_only = _extract_title_from_query(query)
        if title_only and title_only != query:
            logging.debug("Searching with title only: '%s' -> '%s'", query, title_only)
            query = title_only
    
    # Progressive search: try small batch first to quickly catch exact matches
    attempt_sizes = []
    if strategy == "best":
        attempt_sizes = sorted({min(5, n), min(10, n), n})
    else:
        attempt_sizes = [n]

    try:
        import yt_dlp  # type: ignore
        for idx, size in enumerate(attempt_sizes):
            search_term = f"ytsearch{size}:{query}"
            # Use flat extraction to speed up metadata-only search
            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": True}) as ydl:
                info = ydl.extract_info(search_term, download=False)
                if not (info and "entries" in info and info["entries"]):
                    continue

                # Attach the original query so 'best' strategy can use it and filter entries
                entries = []
                for e in info["entries"]:
                    try:
                        e_copy = dict(e)
                    except Exception:
                        e_copy = e
                    if _should_skip_entry(e_copy, query):
                        continue
                    e_copy["__query__"] = query
                    if enriched:
                        e_copy["__spotify__"] = enriched
                    entries.append(e_copy)

                # Fast path: if best strategy, return early on an exact match
                if strategy == "best":
                    from .matching import _find_exact_matches
                    exact = _find_exact_matches(entries, query)
                    if exact:
                        chosen_exact = exact[0]
                        title = chosen_exact.get("title")
                        channel = chosen_exact.get("channel") or chosen_exact.get("uploader")
                        views = chosen_exact.get("view_count")
                        logging.debug("Selected (exact): title='%s' channel='%s' views=%s", title, channel, views)
                        return chosen_exact.get("webpage_url") or chosen_exact.get("url")

                # On the last attempt or for non-best strategies, pick normally
                is_last_attempt = idx == len(attempt_sizes) - 1
                if is_last_attempt or strategy != "best":
                    chosen = pick_entry(entries, strategy)
                    if not chosen:
                        continue
                    title = chosen.get("title")
                    channel = chosen.get("channel") or chosen.get("uploader")
                    views = chosen.get("view_count")
                    logging.debug("Selected: title='%s' channel='%s' views=%s", title, channel, views)
                    return chosen.get("webpage_url") or chosen.get("url")
    except Exception as error:  # noqa: BLE001
        logging.error("Search failed for '%s': %s", query, error)
    return None


def collect_search_urls(query: str, search_count: int) -> List[str]:
    """Return a list of result URLs for a given query (metadata-only, flat extract)."""
    n = max(1, min(int(search_count), 50))
    
    # Apply deep search if enabled
    config = get_runtime_config()
    if config.deep_search:
        n = min(n * 2, 50)  # Double the search count, but cap at 50
    
    # Apply title-only search if enabled
    if config.search_without_authors:
        original_query = query
        title_only = _extract_title_from_query(query)
        if title_only and title_only != query:
            logging.debug("Listing with title only: '%s' -> '%s'", query, title_only)
            query = title_only
    
    search_term = f"ytsearch{n}:{query}"
    urls: List[str] = []
    try:
        import yt_dlp  # type: ignore
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": True}) as ydl:
            info = ydl.extract_info(search_term, download=False)
            if info and "entries" in info and info["entries"]:
                for e in info["entries"]:
                    try:
                        e_copy = dict(e)
                    except Exception:
                        e_copy = e
                    if _should_skip_entry(e_copy, query):
                        continue
                    url = e_copy.get("webpage_url") or e_copy.get("url")
                    if url:
                        urls.append(str(url))
    except Exception as error:  # noqa: BLE001
        logging.error("Search (list) failed for '%s': %s", query, error)
    return urls


__all__ = ["search_video_url", "collect_search_urls"]
