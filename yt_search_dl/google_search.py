"""
Google search integration for YouTube Search to MP3 Downloader.

This module provides Google search enrichment to improve query accuracy
by finding additional context, correct song titles, and artist information.
"""

import logging
import re
from typing import Dict, List, Optional, cast
from urllib.parse import quote_plus

try:
    import requests  # type: ignore[import-untyped]
except ImportError:
    requests = None  # type: ignore

try:
    from bs4 import Tag
except ImportError:
    Tag = None  # type: ignore

from .config import get_runtime_config
from .utils import _normalize_text, _similarity


def _extract_song_info_from_google_results(results: List[Dict]) -> Dict:
    """Extract song information from Google search results."""
    song_info = {"title": "", "artist": "", "album": "", "year": "", "confidence": 0.0}

    # Common patterns for song information
    title_patterns = [
        r'"([^"]+)"\s*(?:by|-\s*|–\s*)\s*([^"]+)',  # "Song Title" by Artist
        r"([^-–]+)\s*(?:by|-\s*|–\s*)\s*([^-–]+)",  # Song Title - Artist
        r"([^(]+)\s*\(([^)]+)\)",  # Song Title (Artist)
    ]

    artist_patterns = [
        r"(?:by|artist|performed by|featuring|feat\.?)\s*([A-Za-z\s&]+)",
        r"([A-Za-z\s&]+)\s*(?:album|song|track|single)",
    ]

    year_patterns = [
        r"\b(19|20)\d{2}\b",  # Years 1900-2099
    ]

    album_patterns = [
        r'(?:from|album|on)\s*["\']([^"\']+)["\']',
        r'(?:album|record)\s*["\']([^"\']+)["\']',
    ]

    total_confidence = 0.0
    count = 0

    for result in results[:5]:  # Check top 5 results
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        text = f"{title} {snippet}"

        # Extract song title and artist
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if not song_info["title"]:
                    song_info["title"] = match.group(1).strip()
                if not song_info["artist"]:
                    song_info["artist"] = match.group(2).strip()
                total_confidence += 0.3
                count += 1
                break

        # Extract artist information
        for pattern in artist_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not song_info["artist"]:
                artist = match.group(1).strip()
                if (
                    len(artist) > 2 and len(artist) < 50
                ):  # Reasonable artist name length
                    song_info["artist"] = artist
                    total_confidence += 0.2
                    count += 1
                    break

        # Extract year
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match and not song_info["year"]:
                year = match.group(0)
                if 1900 <= int(year) <= 2025:  # Reasonable year range
                    song_info["year"] = year
                    total_confidence += 0.1
                    count += 1
                    break

        # Extract album
        for pattern in album_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and not song_info["album"]:
                album = match.group(1).strip()
                if len(album) > 2 and len(album) < 100:  # Reasonable album name length
                    song_info["album"] = album
                    total_confidence += 0.1
                    count += 1
                    break

    if count > 0:
        song_info["confidence"] = float(total_confidence / count)

    return song_info


def _google_search_enrich_query(query: str) -> Optional[Dict]:
    """Enrich query using Google search to find better song information."""
    config = get_runtime_config()
    if not config.use_google_search:
        return None

    api_key = config.google_api_key
    search_engine_id = config.google_search_engine_id

    if not api_key or not search_engine_id:
        logging.warning(
            "Google: --use-google-search set but API credentials are missing"
        )
        return None

    try:
        if requests is None:
            logging.warning("Google: requests library not available")
            return None

        # Prepare search query
        search_query = f'"{query}" song artist music'

        # Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": search_query,
            "num": 5,  # Number of results
            "safe": "active",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        if not items:
            logging.debug("Google: no search results found for '%s'", query)
            return None

        # Extract song information from results
        song_info = _extract_song_info_from_google_results(items)

        if song_info["confidence"] < config.google_min_confidence:
            logging.debug(
                "Google: confidence too low (%.2f < %.2f) for '%s'",
                song_info["confidence"],
                config.google_min_confidence,
                query,
            )
            return None

        # Build enriched query
        enriched_parts = []
        if song_info["title"]:
            enriched_parts.append(song_info["title"])
        if song_info["artist"]:
            enriched_parts.append(song_info["artist"])

        if enriched_parts:
            enriched_query = " ".join(enriched_parts)

            # Check if the enriched query is significantly different
            original_norm = _normalize_text(query)
            enriched_norm = _normalize_text(enriched_query)
            similarity = _similarity(original_norm, enriched_norm)

            if similarity < 0.8:  # Only use if significantly different
                logging.info(
                    "Google: enriched '%s' -> '%s' (confidence=%.2f)",
                    query,
                    enriched_query,
                    song_info["confidence"],
                )
                return {
                    "query": enriched_query,
                    "title": song_info["title"],
                    "artist": song_info["artist"],
                    "album": song_info["album"],
                    "year": song_info["year"],
                    "confidence": song_info["confidence"],
                    "original_query": query,
                }

        return None

    except Exception as error:
        logging.error("Google: search failed for '%s': %s", query, error)
        return None


def _google_search_fallback(query: str) -> Optional[Dict]:
    """Fallback Google search using web scraping (when API is not available)."""
    config = get_runtime_config()
    if not config.use_google_search_fallback:
        return None

    try:
        if requests is None:
            logging.warning("Google (fallback): requests library not available")
            return None

        from bs4 import BeautifulSoup

        # Prepare search query
        search_query = f'"{query}" song artist music'
        encoded_query = quote_plus(search_query)

        # Use a simple web search (note: this is limited and may not work reliably)
        url = f"https://www.google.com/search?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract search results (this is a simplified approach)
        results = []
        for result in soup.find_all("div", class_="g")[:5]:
            # Cast to Tag to access find method
            result_tag = cast(Tag, result)
            title_elem = result_tag.find("h3")
            snippet_elem = result_tag.find("div", class_="VwiC3b")

            if title_elem and snippet_elem:
                results.append(
                    {"title": title_elem.get_text(), "snippet": snippet_elem.get_text()}
                )

        if results:
            song_info = _extract_song_info_from_google_results(results)

            if song_info["confidence"] >= config.google_min_confidence:
                enriched_parts = []
                if song_info["title"]:
                    enriched_parts.append(song_info["title"])
                if song_info["artist"]:
                    enriched_parts.append(song_info["artist"])

                if enriched_parts:
                    enriched_query = " ".join(enriched_parts)
                    logging.info(
                        "Google (fallback): enriched '%s' -> '%s'",
                        query,
                        enriched_query,
                    )
                    return {
                        "query": enriched_query,
                        "title": song_info["title"],
                        "artist": song_info["artist"],
                        "confidence": song_info["confidence"],
                        "original_query": query,
                    }

        return None

    except Exception as error:
        logging.debug("Google (fallback): search failed for '%s': %s", query, error)
        return None


def google_search_enrich_query(query: str) -> Optional[Dict]:
    """Main function to enrich query using Google search."""
    # Try API first
    enriched = _google_search_enrich_query(query)
    if enriched:
        return enriched

    # Fallback to web scraping if API fails
    return _google_search_fallback(query)


def google_search_filter_query(query: str) -> Optional[str]:
    """Filter a query through Google search and return the first result title as the new query."""
    config = get_runtime_config()
    if not config.filter_queries_with_google:
        logging.debug("Google filter: feature not enabled")
        return None

    # Check if API credentials are available
    if not config.google_api_key or not config.google_search_engine_id:
        logging.debug(
            "Google filter: API credentials not configured (key: %s, engine: %s)",
            "set" if config.google_api_key else "missing",
            "set" if config.google_search_engine_id else "missing",
        )
        return None

    try:
        if requests is None:
            logging.warning(
                "Google: requests library not available for query filtering"
            )
            return None

        # Prepare search query - search for the query without quotes for broader results
        search_query = query

        # Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.google_api_key,
            "cx": config.google_search_engine_id,
            "q": search_query,
            "num": 1,  # Only get the first result
            "safe": "active",
        }

        logging.debug("Google filter: searching for '%s' with API", search_query)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        if not items:
            logging.debug("Google filter: no search results found for '%s'", query)
            return None

        # Get the first result title
        first_result = items[0]
        result_title = first_result.get("title", "").strip()

        if result_title:
            logging.info("Google filter: '%s' -> '%s'", query, result_title)
            return result_title

        logging.debug("Google filter: empty title in first result for '%s'", query)
        return None

    except Exception as error:
        logging.error("Google filter: search failed for '%s': %s", query, error)
        return None


def google_search_filter_query_fallback(query: str) -> Optional[str]:
    """Fallback Google search filtering using web scraping."""
    config = get_runtime_config()
    if not config.filter_queries_with_google:
        return None

    # Try LLM parsing first if enabled
    if config.use_llm_google_parsing:
        llm_result = _google_search_filter_with_llm(query)
        if llm_result:
            return llm_result

    try:
        if requests is None:
            logging.warning("Google filter (fallback): requests library not available")
            return None

        from bs4 import BeautifulSoup

        # Prepare search query - without quotes for broader results
        search_query = query
        encoded_query = quote_plus(search_query)

        # Use a simple web search
        url = f"https://www.google.com/search?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        logging.debug(
            "Google filter (fallback): searching for '%s' with web scraping",
            search_query,
        )
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract first search result title - try multiple approaches
        title_elem = None

        # First, try to find search result containers
        search_results = soup.find_all("div", class_="g") or soup.find_all(
            "div", {"data-sokoban-container": True}
        )

        if search_results:
            # Look for titles within the first search result
            first_result = search_results[0]
            # Cast to Tag to access find method
            first_result_tag = cast(Tag, first_result)
            title_elem = first_result_tag.find("h3") or first_result_tag.find(
                "a", {"data-ved": True}
            )
            if title_elem:
                logging.debug(
                    "Google filter (fallback): found title in search result container"
                )

        # If no results found in containers, try direct selectors
        if not title_elem:
            selectors = [
                ".LC20lb",  # Modern Google title class (most reliable)
                "h3",  # Standard title
                "a[data-ved]",  # Link with data-ved attribute
                ".DKV0Md",  # Alternative title class
                "a[href*='http']:not([href*='google.com'])",  # Any link with http but not Google
            ]

            for selector in selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    logging.debug(
                        "Google filter (fallback): found title with selector '%s'",
                        selector,
                    )
                    break

        if title_elem:
            result_title = title_elem.get_text().strip()
            logging.debug(
                "Google filter (fallback): found title text: '%s' (length: %d)",
                result_title,
                len(result_title),
            )
            if result_title and len(result_title) > 5:  # Ensure it's a meaningful title
                logging.info(
                    "Google filter (fallback): '%s' -> '%s'", query, result_title
                )
                return result_title
            else:
                logging.debug("Google filter (fallback): title too short or empty")

        logging.debug("Google filter (fallback): no results found for '%s'", query)
        return None

    except Exception as error:
        logging.debug(
            "Google filter (fallback): search failed for '%s': %s", query, error
        )
        return None


def _google_search_filter_with_llm(query: str) -> Optional[str]:
    """Use LLM to parse Google search results and extract the first result title."""
    config = get_runtime_config()

    if not config.use_llm_google_parsing or not config.llm_api_key:
        return None

    try:
        if requests is None:
            logging.warning("Google filter (LLM): requests library not available")
            return None

        from bs4 import BeautifulSoup

        # Prepare search query
        search_query = query
        encoded_query = quote_plus(search_query)

        # Get Google search results
        url = f"https://www.google.com/search?q={encoded_query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        logging.debug("Google filter (LLM): searching for '%s'", search_query)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the HTML content for LLM analysis
        html_content = soup.prettify()[
            :8000
        ]  # Limit to first 8KB to avoid token limits

        # Prepare prompt for LLM
        prompt = f"""
You are analyzing a Google search results page for the query: "{query}"

Extract the title of the FIRST search result (not ads, not navigation links, not Google's own pages).

HTML content:
{html_content}

Instructions:
1. Look for the first actual search result title
2. Ignore Google navigation, ads, and internal links
3. Return ONLY the title text, nothing else
4. If no clear search result is found, return "NO_RESULT"

Return the title of the first search result:
"""

        # Call LLM API
        llm_result = _call_llm_api(prompt, config.llm_api_key, config.llm_model)

        if llm_result and llm_result.strip() != "NO_RESULT":
            result_title = llm_result.strip()
            if len(result_title) > 5:  # Ensure it's meaningful
                logging.info("Google filter (LLM): '%s' -> '%s'", query, result_title)
                return result_title

        logging.debug("Google filter (LLM): no valid result found for '%s'", query)
        return None

    except Exception as error:
        logging.error("Google filter (LLM): failed for '%s': %s", query, error)
        return None


def _call_llm_api(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Call LLM API (OpenAI, local LLMs, or compatible)."""
    config = get_runtime_config()

    try:
        # Try local LLM first if base URL is configured
        if config.llm_base_url:
            return _call_local_llm_api(prompt, model, config.llm_base_url)

        # Try OpenAI if API key is provided
        elif api_key and "gpt" in model.lower():
            return _call_openai_api(prompt, api_key, model)

        # Add other LLM providers here as needed
        else:
            logging.warning(
                "LLM model '%s' not supported or no credentials provided", model
            )
            return None
    except Exception as error:
        logging.error("LLM API call failed: %s", error)
        return None


def _call_openai_api(prompt: str, api_key: str, model: str) -> Optional[str]:
    """Call OpenAI API."""
    try:
        import openai  # type: ignore[import-not-found]

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts information from HTML content.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    except ImportError:
        logging.error("OpenAI library not installed. Install with: pip install openai")
        return None
    except Exception as error:
        logging.error("OpenAI API call failed: %s", error)
        return None


def _call_local_llm_api(prompt: str, model: str, base_url: str) -> Optional[str]:
    """Call local LLM API (Ollama, LM Studio, etc.)."""
    try:
        import requests

        # Prepare the request payload for local LLM
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts information from HTML content.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }

        # Make the API call
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        return data.get("message", {}).get("content", "").strip()

    except ImportError:
        logging.error("Requests library not available for local LLM")
        return None
    except Exception as error:
        logging.error("Local LLM API call failed: %s", error)
        return None


def google_search_filter_query_main(query: str) -> Optional[str]:
    """Main function to filter query through Google search."""
    logging.debug("Google filter (main): processing query '%s'", query)

    # Try API first
    filtered = google_search_filter_query(query)
    if filtered:
        logging.debug("Google filter (main): API method succeeded")
        return filtered

    # Fallback to web scraping if API fails
    logging.debug("Google filter (main): trying fallback method")
    fallback_result = google_search_filter_query_fallback(query)
    if fallback_result:
        logging.debug("Google filter (main): fallback method succeeded")
        return fallback_result

    logging.debug("Google filter (main): both methods failed, returning None")
    return None


__all__ = [
    "google_search_enrich_query",
    "google_search_filter_query_main",
]
