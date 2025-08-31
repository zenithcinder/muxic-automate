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


def _parse_query_fallback(query: str) -> Optional[Dict]:
    """Parse query to extract song title and artist information."""
    logging.debug("Google (fallback): parsing query fallback for '%s'", query)
    
    # Common separators for song title and artist
    separators = [
        (" - ", "Title - Artist"),
        (" – ", "Title – Artist"), 
        (" by ", "Title by Artist"),
        (" feat ", "Title feat Artist"),
        (" ft ", "Title ft Artist"),
        (" featuring ", "Title featuring Artist"),
        (" / ", "Title / Artist"),
        (" | ", "Title | Artist"),
    ]
    
    for separator, pattern_name in separators:
        if separator in query:
            parts = query.split(separator, 1)
            if len(parts) == 2:
                part1, part2 = parts[0].strip(), parts[1].strip()
                
                # Determine which part is likely the title vs artist
                # Usually the format is "Title - Artist" or "Artist - Title"
                # We'll assume the longer part is the title
                if len(part1) > len(part2):
                    title, artist = part1, part2
                else:
                    title, artist = part2, part1
                
                if len(title) > 3 and len(artist) > 2:
                    logging.info(
                        "Google (fallback): parsed query '%s' using pattern '%s' -> title: '%s', artist: '%s'",
                        query, pattern_name, title, artist
                    )
                    return {
                        "query": f"{title} {artist}",
                        "title": title,
                        "artist": artist,
                        "confidence": 0.6,  # Medium confidence for parsed query
                        "original_query": query,
                    }
    
    # If no separators found, try to split on common words
    common_words = ["song", "track", "music", "single", "album"]
    for word in common_words:
        if word.lower() in query.lower():
            # Split on the word and take the parts
            parts = query.lower().split(word.lower(), 1)
            if len(parts) == 2:
                part1, part2 = parts[0].strip(), parts[1].strip()
                if len(part1) > 3 and len(part2) > 3:
                    logging.info(
                        "Google (fallback): parsed query '%s' using word '%s' -> part1: '%s', part2: '%s'",
                        query, word, part1, part2
                    )
                    return {
                        "query": f"{part1} {part2}",
                        "title": part1.title(),
                        "artist": part2.title(),
                        "confidence": 0.4,  # Lower confidence for word-based parsing
                        "original_query": query,
                    }
    
    # If all else fails, try to split on spaces and assume first part is title, rest is artist
    words = query.split()
    if len(words) >= 3:
        # For longer queries, try to find a better split point
        # Look for common artist name patterns (e.g., "Led Zeppelin", "The Beatles")
        for i in range(1, min(4, len(words))):
            potential_title = " ".join(words[:i])
            potential_artist = " ".join(words[i:])
            
            # Check if this split makes sense
            if (len(potential_title) > 2 and len(potential_artist) > 2 and
                not any(word.lower() in ["the", "and", "or", "feat", "ft", "featuring"] for word in words[i:i+1])):
                
                logging.info(
                    "Google (fallback): parsed query '%s' using smart word splitting -> title: '%s', artist: '%s'",
                    query, potential_title, potential_artist
                )
                return {
                    "query": f"{potential_title} {potential_artist}",
                    "title": potential_title,
                    "artist": potential_artist,
                    "confidence": 0.4,  # Slightly higher confidence for smart splitting
                    "original_query": query,
                }
        
        # Fallback to simple splitting
        title_words = words[:2] if len(words) >= 4 else words[:1]
        artist_words = words[len(title_words):]
        
        if len(title_words) > 0 and len(artist_words) > 0:
            title = " ".join(title_words)
            artist = " ".join(artist_words)
            
            if len(title) > 2 and len(artist) > 2:
                logging.info(
                    "Google (fallback): parsed query '%s' using basic word splitting -> title: '%s', artist: '%s'",
                    query, title, artist
                )
                return {
                    "query": f"{title} {artist}",
                    "title": title,
                    "artist": artist,
                    "confidence": 0.3,  # Lower confidence for basic splitting
                    "original_query": query,
                }
    
    logging.debug("Google (fallback): could not parse query '%s'", query)
    return None


def _extract_song_info_from_google_results(results: List[Dict]) -> Dict:
    """Extract song information from Google search results."""
    song_info = {"title": "", "artist": "", "album": "", "year": "", "confidence": 0.0}

    # Common patterns for song information
    title_patterns = [
        r'"([^"]+)"\s*(?:by|-\s*|–\s*)\s*([^"]+)',  # "Song Title" by Artist
        r"([^-–]+)\s*(?:by|-\s*|–\s*)\s*([^-–]+)",  # Song Title - Artist
        r"([^(]+)\s*\(([^)]+)\)",  # Song Title (Artist)
        r"([^-–\s]+(?:\s+[^-–\s]+)*)\s*(?:by|-\s*|–\s*)\s*([^-–\s]+(?:\s+[^-–\s]+)*)",  # More flexible pattern
    ]

    artist_patterns = [
        r"(?:by|artist|performed by|featuring|feat\.?)\s*([A-Za-z\s&]+)",
        r"([A-Za-z\s&]+)\s*(?:album|song|track|single)",
        r"(?:artist|performer|band)\s*[:\-]\s*([A-Za-z\s&]+)",  # Artist: Name format
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
                total_confidence += 0.4  # Increased confidence for title/artist match
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
        
        # Boost confidence if we have basic title/artist info
        current_confidence = song_info.get("confidence", 0.0)
        if isinstance(current_confidence, (int, float)):
            confidence_val = float(current_confidence)
        else:
            confidence_val = 0.0
            
        if song_info.get("title") and song_info.get("artist"):
            song_info["confidence"] = min(0.8, confidence_val + 0.2)
        elif song_info.get("title") or song_info.get("artist"):
            song_info["confidence"] = min(0.7, confidence_val + 0.1)

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
    logging.debug("Google (fallback): called with query '%s', fallback enabled: %s", 
                 query, config.use_google_search_fallback)
    
    if not config.use_google_search_fallback:
        logging.debug("Google (fallback): fallback disabled in config")
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Check if we got a valid HTML response
        if not response.text or len(response.text) < 1000:
            logging.warning("Google (fallback): response too short, likely blocked or invalid")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Debug: log some basic info about the response
        logging.debug("Google (fallback): response length: %d, contains 'google': %s", 
                     len(response.text), 'google' in response.text.lower())
        
        # Debug: log some HTML structure info
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            # Only do this expensive operation if debug logging is enabled
            try:
                all_divs = soup.find_all("div")
                div_classes = []
                for div in all_divs[:20]:
                    if hasattr(div, 'get') and isinstance(div, Tag):
                        class_attr = div.get("class")
                        if class_attr:
                            div_classes.append(class_attr)
                logging.debug("Google (fallback): first 20 div classes: %s", div_classes[:10])
            except Exception as e:
                logging.debug("Google (fallback): error analyzing HTML structure: %s", e)

        # Extract search results using multiple selector strategies
        results = []
        
        # Strategy 1: Look for common Google result containers
        selectors = [
            "div[data-sokoban-container]",  # Modern Google results
            "div.g",  # Classic Google results
            "div[jscontroller]",  # JavaScript controlled results
            "div[data-ved]",  # Results with view data
            "div[class*='g']",  # Any div with class containing 'g'
            "div[class*='result']",  # Any div with class containing 'result'
        ]
        
        logging.debug("Google (fallback): trying selectors: %s", selectors)
        
        for selector in selectors:
            potential_results = soup.select(selector)
            logging.debug("Google (fallback): selector '%s' found %d results", selector, len(potential_results))
            if potential_results:
                for result in potential_results[:10]:  # Check more results
                    # Look for title in various possible locations
                    title_elem = (
                        result.find("h3") or 
                        result.find("a", href=True) or
                        result.find("div", {"role": "heading"}) or
                        result.find("span", {"role": "heading"})
                    )
                    
                    # Look for snippet in various possible locations
                    snippet_elem = (
                        result.find("div", class_="VwiC3b") or  # Try original class first
                        result.find("div", class_="LC20lb") or   # Alternative class
                        result.find("span", class_="aCOpRe") or  # Another possible class
                        result.find("div", {"data-content-feature": "1"}) or  # Content feature
                        result.find("div", string=re.compile(r".{20,}"))  # Any div with substantial text
                    )
                    
                    if title_elem and snippet_elem:
                        title_text = title_elem.get_text().strip()
                        snippet_text = snippet_elem.get_text().strip()
                        
                        logging.debug("Google (fallback): found result - title: '%s', snippet: '%s'", 
                                   title_text[:50], snippet_text[:50])
                        
                        # Filter out results that are too short or likely not music-related
                        if (len(title_text) > 10 and len(snippet_text) > 20 and
                            any(keyword in title_text.lower() or keyword in snippet_text.lower() 
                                for keyword in ["song", "music", "artist", "album", "track", "single"])):
                            results.append({
                                "title": title_text,
                                "snippet": snippet_text
                            })
                            logging.debug("Google (fallback): added result %d", len(results))
                            if len(results) >= 5:  # Limit to top 5 results
                                break
                if results:
                    break  # If we found results with this selector, stop trying others

        # Strategy 2: Try to extract info from page title and meta tags if no results found
        if not results:
            logging.debug("Google (fallback): no results from selectors, trying page title/meta approach")
            
            # Try to get page title
            page_title = soup.find("title")
            if page_title:
                title_text = page_title.get_text().strip()
                logging.debug("Google (fallback): page title: '%s'", title_text)
                
                # If page title contains the query, it might be useful
                if query.lower() in title_text.lower():
                    # Try to extract song info from the title
                    if " - " in title_text or " by " in title_text.lower():
                        parts = re.split(r'\s*[-–]\s*|\s+by\s+', title_text, flags=re.IGNORECASE)
                        if len(parts) >= 2:
                            potential_title = parts[0].strip()
                            potential_artist = parts[1].strip()
                            
                            if len(potential_title) > 3 and len(potential_artist) > 2:
                                logging.info(
                                    "Google (fallback): extracted from page title: '%s' -> '%s'",
                                    title_text, f"{potential_title} {potential_artist}"
                                )
                                return {
                                    "query": f"{potential_title} {potential_artist}",
                                    "title": potential_title,
                                    "artist": potential_artist,
                                    "confidence": 0.6,
                                    "original_query": query,
                                }
            
            # Strategy 3: Look for any text that might contain song information
            logging.debug("Google (fallback): trying to find any music-related text")
            all_text = soup.get_text()
            
            # Look for patterns like "Song Title - Artist" in the page text
            music_patterns = [
                r'([A-Z][A-Za-z\s&\']+)\s*[-–]\s*([A-Z][A-Za-z\s&\']+)',  # Title - Artist
                r'([A-Z][A-Za-z\s&\']+)\s+by\s+([A-Z][A-Za-z\s&\']+)',   # Title by Artist
                r'"([^"]+)"\s*(?:by|-\s*|–\s*)\s*([^"]+)',                # "Title" by Artist
            ]
            
            for pattern in music_patterns:
                matches = re.findall(pattern, all_text)
                for match in matches[:3]:  # Check first 3 matches
                    if len(match) == 2:
                        potential_title, potential_artist = match[0].strip(), match[1].strip()
                        
                        # Filter out very short or very long matches
                        if (3 < len(potential_title) < 50 and 
                            3 < len(potential_artist) < 50 and
                            not any(word in potential_title.lower() for word in ["google", "search", "results"]) and
                            not any(word in potential_artist.lower() for word in ["google", "search", "results"])):
                            
                            logging.info(
                                "Google (fallback): found music pattern: '%s' - '%s'",
                                potential_title, potential_artist
                            )
                            return {
                                "query": f"{potential_title} {potential_artist}",
                                "title": potential_title,
                                "artist": potential_artist,
                                "confidence": 0.5,
                                "original_query": query,
                            }

        if results:
            logging.debug("Google (fallback): found %d results", len(results))
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
                        "Google (fallback): enriched '%s' -> '%s' (confidence=%.2f)",
                        query,
                        enriched_query,
                        song_info["confidence"],
                    )
                    return {
                        "query": enriched_query,
                        "title": song_info["title"],
                        "artist": song_info["artist"],
                        "confidence": song_info["confidence"],
                        "original_query": query,
                    }
            else:
                logging.debug(
                    "Google (fallback): confidence too low (%.2f < %.2f) for '%s'",
                    song_info["confidence"],
                    config.google_min_confidence,
                    query,
                )
        else:
            logging.debug("Google (fallback): no results found for '%s'", query)
            
                    # Debug: log some HTML content if no results found
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            try:
                # Log a small portion of the HTML to help debug
                html_sample = response.text[:2000] if len(response.text) > 2000 else response.text
                logging.debug("Google (fallback): HTML sample: %s", html_sample)
            except Exception as e:
                logging.debug("Google (fallback): error logging HTML sample: %s", e)
        
        # Since Google is returning JavaScript-heavy pages, we'll rely on query parsing
        logging.debug("Google (fallback): Google returned JavaScript-heavy page, using query parsing fallback")
        
        # Enhanced fallback: try to extract basic info from the query itself
        return _parse_query_fallback(query)

        return None

    except Exception as error:
        logging.debug("Google (fallback): search failed for '%s': %s", query, error)
        
        # Try alternative search approach - search for the query directly
        try:
            logging.debug("Google (fallback): trying alternative search approach")
            
            # Use the enhanced query parsing function
            fallback_result = _parse_query_fallback(query)
            if fallback_result:
                return fallback_result
                
        except Exception as fallback_error:
            logging.debug("Google (fallback): alternative approach also failed: %s", fallback_error)
        
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

        # Prepare search query - add music context for better results
        search_query = f'{query} song music'
        
        # Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.google_api_key,
            "cx": config.google_search_engine_id,
            "q": search_query,
            "num": 5,  # Get more results to find the best one
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

        # Process and validate results to find the best one
        best_result = _find_best_google_result(query, items)
        
        if best_result:
            logging.info("Google filter: '%s' -> '%s' (confidence=%.2f)", 
                        query, best_result["title"], best_result["confidence"])
            return best_result["title"]

        logging.debug("Google filter: no valid results found for '%s'", query)
        return None

    except Exception as error:
        logging.error("Google filter: search failed for '%s': %s", query, error)
        return None


def _find_best_google_result(original_query: str, search_results: List[Dict]) -> Optional[Dict]:
    """Find the best Google search result by analyzing and scoring each result."""
    config = get_runtime_config()
    
    if not search_results:
        return None
    
    scored_results = []
    
    for i, result in enumerate(search_results):
        title = result.get("title", "").strip()
        snippet = result.get("snippet", "").strip()
        link = result.get("link", "")
        source = result.get("source", "unknown")
        
        if not title or len(title) < 5:
            continue
            
        # Score the result based on multiple factors
        score = _score_google_result(original_query, title, snippet, link, i)
        
        logging.debug("Google scoring: '%s' (source: %s) -> score: %.1f", title, source, score)
        
        if score > 0:  # Only include results with positive scores
            scored_results.append({
                "title": title,
                "snippet": snippet,
                "link": link,
                "score": score,
                "confidence": min(0.9, score / 100.0),  # Normalize to 0-0.9 range
                "source": source
            })
    
    if not scored_results:
        logging.debug("Google scoring: no results with positive scores")
        return None
    
    # Sort by score (highest first) and return the best result
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    best_result = scored_results[0]
    
    # Use different thresholds based on source
    min_score = config.google_filter_min_score
    if best_result.get("source") in ["duckduckgo", "bing", "pattern_match"]:
        # Lower threshold for alternative sources
        min_score = max(15.0, min_score * 0.5)
        logging.debug("Google scoring: using lower threshold %.1f for source %s", min_score, best_result.get("source"))
    
    logging.debug("Google scoring: best result '%s' has score %.1f (threshold: %.1f)", 
                best_result["title"], best_result["score"], min_score)
    
    # Only return if the result is significantly better than the original query
    if best_result["score"] >= min_score:
        return best_result
    
    logging.debug("Google scoring: best result score %.1f below threshold %.1f", best_result["score"], min_score)
    return None


def _score_google_result(original_query: str, title: str, snippet: str, link: str, position: int) -> float:
    """Score a Google search result based on relevance and quality."""
    config = get_runtime_config()
    score = 0.0
    
    # Base score based on position (first results get higher scores)
    score += max(0, 50 - (position * 10))
    
    # Boost for music-related content (if enabled)
    if config.google_filter_boost_music_keywords:
        music_keywords = ["song", "music", "artist", "album", "track", "single", "lyrics", "official"]
        title_lower = title.lower()
        snippet_lower = snippet.lower()
        
        for keyword in music_keywords:
            if keyword in title_lower:
                score += 15
            if keyword in snippet_lower:
                score += 10
    
    # Boost for video platforms (likely music videos) - if enabled
    if config.google_filter_prefer_video_platforms:
        video_domains = ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]
        for domain in video_domains:
            if domain in link.lower():
                score += 20
    
    # Penalize for common unwanted content - if enabled
    if config.google_filter_penalize_spam:
        unwanted_patterns = [
            r"google\.com", r"search", r"results", r"ads", r"advertisement",
            r"sponsored", r"promoted", r"buy", r"download", r"free",
            r"mp3", r"mp4", r"audio", r"video"  # Avoid generic file format results
        ]
        
        for pattern in unwanted_patterns:
            if re.search(pattern, title.lower(), re.IGNORECASE):
                score -= 25
    
    # Penalize for very long titles (often spam)
    if len(title) > 100:
        score -= 20
    elif len(title) > 60:
        score -= 10
    
    # Penalize for very short titles (often not descriptive)
    if len(title) < 10:
        score -= 15
    
    # Boost for similarity to original query
    original_words = set(original_query.lower().split())
    title_words = set(title.lower().split())
    
    # Calculate word overlap
    common_words = original_words.intersection(title_words)
    if common_words:
        overlap_ratio = len(common_words) / len(original_words)
        score += overlap_ratio * 30
    
    # Penalize for completely unrelated content
    if not common_words and len(original_words) > 2:
        score -= 40
    
    # Boost for proper capitalization (indicates quality content)
    if title[0].isupper() and any(word[0].isupper() for word in title.split() if len(word) > 2):
        score += 10
    
    # Penalize for all caps (often spam)
    if title.isupper():
        score -= 30
    
    # Penalize for excessive punctuation or special characters
    special_char_count = sum(1 for c in title if c in "!@#$%^&*()_+-=[]{}|;':\",./<>?")
    if special_char_count > 5:
        score -= special_char_count * 2
    
    return max(0, score)  # Ensure non-negative score


def _try_alternative_search_engines(query: str) -> List[Dict]:
    """Try alternative search engines when Google fails."""
    results = []
    
    # Try DuckDuckGo (more lenient with automated requests)
    try:
        ddg_url = f"https://duckduckgo.com/html/?q={quote_plus(query + ' song music')}"
        ddg_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        response = requests.get(ddg_url, headers=ddg_headers, timeout=10)
        if response.status_code == 200 and len(response.text) > 1000:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # DuckDuckGo result selectors
            ddg_results = soup.select(".result__title, .result__a, .web-result__title")
            for result in ddg_results[:5]:
                title_text = result.get_text().strip()
                if len(title_text) > 10 and len(title_text) < 200:
                    results.append({
                        "title": title_text,
                        "snippet": f"DuckDuckGo result: {title_text}",
                        "link": "",
                        "source": "duckduckgo"
                    })
                    
    except Exception as error:
        logging.debug("Alternative search: DuckDuckGo failed: %s", error)
    
    # Try Bing (sometimes more lenient)
    try:
        bing_url = f"https://www.bing.com/search?q={quote_plus(query + ' song music')}"
        bing_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        response = requests.get(bing_url, headers=bing_headers, timeout=10)
        if response.status_code == 200 and len(response.text) > 1000:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Bing result selectors
            bing_results = soup.select("h2 a, .b_title a, .b_algo h2 a")
            for result in bing_results[:5]:
                title_text = result.get_text().strip()
                if len(title_text) > 10 and len(title_text) < 200:
                    results.append({
                        "title": title_text,
                        "snippet": f"Bing result: {title_text}",
                        "link": "",
                        "source": "bing"
                    })
                    
    except Exception as error:
        logging.debug("Alternative search: Bing failed: %s", error)
    
    return results


def _extract_google_search_results_from_html(soup, query: str) -> List[Dict]:
    """Extract search results from Google HTML page with improved selectors."""
    results = []
    
    # Updated selectors for current Google HTML structure
    selectors = [
        # Modern Google results (2024+)
        "div[data-sokoban-container] div[data-ved]",
        "div[jscontroller] div[data-ved]",
        "div[class*='g'] div[data-ved]",
        
        # Alternative modern selectors
        "div[data-hveid]",
        "div[class*='tF2Cxc']",
        "div[class*='yuRUbf']",
        
        # Classic Google results (fallback)
        "div.g",
        "div[class*='result']",
        "div[class*='search']",
        
        # Generic result containers
        "div[role='main'] div[class*='g']",
        "div#search div[class*='g']",
    ]
    
    for selector in selectors:
        try:
            potential_results = soup.select(selector)
            if potential_results:
                logging.debug("Google parser: found %d results with selector '%s'", len(potential_results), selector)
                
                for result in potential_results[:10]:
                    # Try multiple title extraction methods
                    title_elem = (
                        result.find("h3") or 
                        result.find("a", href=True) or
                        result.find("div", {"role": "heading"}) or
                        result.find("span", {"role": "heading"}) or
                        result.find("div", class_="LC20lb") or
                        result.find("div", class_="DKV0Md") or
                        result.find("div", class_="title") or
                        result.find("a", class_="title")
                    )
                    
                    # Try multiple snippet extraction methods
                    snippet_elem = (
                        result.find("div", class_="VwiC3b") or
                        result.find("div", class_="LC20lb") or
                        result.find("span", class_="aCOpRe") or
                        result.find("div", {"data-content-feature": "1"}) or
                        result.find("div", class_="snippet") or
                        result.find("div", class_="description") or
                        result.find("div", string=re.compile(r".{20,}"))
                    )
                    
                    link_elem = result.find("a", href=True)
                    
                    if title_elem:
                        title_text = title_elem.get_text().strip()
                        snippet_text = snippet_elem.get_text().strip() if snippet_elem else ""
                        link_text = link_elem.get("href", "") if link_elem else ""
                        
                        # Enhanced validation
                        if (len(title_text) > 5 and len(title_text) < 200 and
                            not any(blocked in title_text.lower() for blocked in [
                                "google", "search", "results", "captcha", "blocked", "robot"
                            ])):
                            
                            # Clean up the title
                            title_text = re.sub(r'\s+', ' ', title_text)  # Remove extra whitespace
                            title_text = re.sub(r'[^\w\s\-–&()[\]]', '', title_text)  # Keep only safe characters
                            
                            results.append({
                                "title": title_text,
                                "snippet": snippet_text[:200] if snippet_text else "",  # Limit snippet length
                                "link": link_text,
                                "source": "google"
                            })
                            
                            if len(results) >= 8:  # Get more results for better scoring
                                break
                                
                if results:
                    logging.debug("Google parser: successfully extracted %d results with selector '%s'", len(results), selector)
                    break
                    
        except Exception as error:
            logging.debug("Google parser: error with selector '%s': %s", selector, error)
            continue
    
    # If no results found with selectors, try alternative approach
    if not results:
        logging.debug("Google parser: no results with selectors, trying text pattern matching")
        
        # Look for any text that might contain song information
        all_text = soup.get_text()
        
        # Look for patterns like "Song Title - Artist" in the page text
        music_patterns = [
            r'([A-Z][A-Za-z\s&\'()[\]]+)\s*[-–]\s*([A-Z][A-Za-z\s&\'()[\]]+)',  # Title - Artist
            r'([A-Z][A-Za-z\s&\'()[\]]+)\s+by\s+([A-Z][A-Za-z\s&\'()[\]]+)',   # Title by Artist
            r'"([^"]+)"\s*(?:by|-\s*|–\s*)\s*([^"]+)',                # "Title" by Artist
        ]
        
        for pattern in music_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches[:3]:  # Check first 3 matches
                if len(match) == 2:
                    potential_title, potential_artist = match[0].strip(), match[1].strip()
                    
                    # Filter out very short or very long matches
                    if (3 < len(potential_title) < 50 and 
                        3 < len(potential_artist) < 50 and
                        not any(word in potential_title.lower() for word in ["google", "search", "results"]) and
                        not any(word in potential_artist.lower() for word in ["google", "search", "results"])):
                        
                        results.append({
                            "title": f"{potential_title} - {potential_artist}",
                            "snippet": f"Found in page content: {potential_title} by {potential_artist}",
                            "link": "",
                            "source": "pattern_match"
                        })
                        
                        if len(results) >= 3:  # Limit pattern-based results
                            break
            if results:
                break
    
    logging.debug("Google parser: total results extracted: %d", len(results))
    return results


def _try_browser_based_google_search(query: str) -> List[Dict]:
    """Try browser-based Google search using Selenium to bypass restrictions."""
    results: List[Dict] = []
    
    try:
        # Try to import Selenium
        try:
            from selenium import webdriver  # type: ignore[import-untyped]
            from selenium.webdriver.common.by import By  # type: ignore[import-untyped]
            from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import-untyped]
            from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import-untyped]
            from selenium.webdriver.chrome.options import Options  # type: ignore[import-untyped]
            from selenium.webdriver.firefox.options import Options as FirefoxOptions  # type: ignore[import-untyped]
        except ImportError:
            logging.debug("Browser search: Selenium not installed. Install with: pip install selenium")
            return results
        
        # Prepare search query
        search_query = f'{query} song music'
        encoded_query = quote_plus(search_query)
        url = f"https://www.google.com/search?q={encoded_query}"
        
        logging.debug("Browser search: attempting browser-based search for '%s'", search_query)
        
        # Try Chrome first, then Firefox
        driver = None
        for browser_type in ['chrome', 'firefox']:
            try:
                if browser_type == 'chrome':
                    options = Options()
                    options.add_argument('--headless')  # Run in background
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--disable-gpu')
                    options.add_argument('--window-size=1920,1080')
                    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                    
                    driver = webdriver.Chrome(options=options)
                else:
                    options = FirefoxOptions()
                    options.add_argument('--headless')
                    options.add_argument('--width=1920')
                    options.add_argument('--height=1080')
                    
                    driver = webdriver.Firefox(options=options)
                
                logging.debug("Browser search: using %s browser", browser_type)
                break
                
            except Exception as e:
                logging.debug("Browser search: failed to start %s: %s", browser_type, e)
                continue
        
        if not driver:
            logging.debug("Browser search: no browser drivers available")
            return results
        
        try:
            # Set page load timeout
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(10)
            
            # Navigate to Google search
            logging.debug("Browser search: navigating to %s", url)
            driver.get(url)
            
            # Wait for search results to load
            wait = WebDriverWait(driver, 15)
            
            # Try multiple selectors for search results
            result_selectors = [
                "div[data-sokoban-container]",
                "div.g",
                "div[jscontroller]",
                "div[data-ved]",
                "div[class*='g']",
                "div[class*='result']",
                "div[class*='tF2Cxc']",  # Modern Google result container
                "div[class*='yuRUbf']",   # Another modern container
                "div[class*='rc']",       # Classic result container
                "div[class*='bkWMgd']",   # Result wrapper
            ]
            
            search_results = []
            for selector in result_selectors:
                try:
                    # Wait for results to appear
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    results_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    if results_elements:
                        logging.debug("Browser search: found %d results with selector '%s'", len(results_elements), selector)
                        search_results = results_elements
                        break
                except Exception as e:
                    logging.debug("Browser search: selector '%s' failed: %s", selector, e)
                    continue
            
            if not search_results:
                logging.debug("Browser search: no search results found with any selector")
                # Debug: show page source to understand structure
                try:
                    page_source = driver.page_source
                    logging.debug("Browser search: page source preview (first 1000 chars): %s", page_source[:1000])
                except:
                    pass
                return results
            
            # Extract results
            for i, result in enumerate(search_results[:10]):
                try:
                    # Try multiple title extraction methods
                    title_elem = None
                    for title_selector in [
                        'h3', 
                        'a[href]', 
                        'div[role="heading"]', 
                        'span[role="heading"]',
                        'div[class*="LC20lb"]',  # Modern Google title class
                        'div[class*="DKV0Md"]',  # Alternative title class
                        'div[class*="title"]',   # Generic title class
                        'a[class*="title"]',     # Title in link
                        'div[class*="heading"]', # Generic heading
                        'span[class*="title"]',  # Title in span
                    ]:
                        try:
                            title_elem = result.find_element(By.CSS_SELECTOR, title_selector)
                            if title_elem and title_elem.text.strip():
                                break
                        except:
                            continue
                    
                    if not title_elem or not title_elem.text.strip():
                        continue
                    
                    title_text = title_elem.text.strip()
                    
                    # Try to get snippet
                    snippet_text = ""
                    for snippet_selector in ['div.VwiC3b', 'div.LC20lb', 'span.aCOpRe', 'div[data-content-feature="1"]']:
                        try:
                            snippet_elem = result.find_element(By.CSS_SELECTOR, snippet_selector)
                            if snippet_elem and snippet_elem.text.strip():
                                snippet_text = snippet_elem.text.strip()
                                break
                        except:
                            continue
                    
                    # Try to get link
                    link_text = ""
                    try:
                        link_elem = result.find_element(By.CSS_SELECTOR, 'a[href]')
                        if link_elem:
                            link_text = link_elem.get_attribute('href') or ""
                    except:
                        pass
                    
                    # Validate result
                    if (len(title_text) > 5 and len(title_text) < 200 and
                        not any(blocked in title_text.lower() for blocked in [
                            "google", "search", "results", "captcha", "blocked", "robot"
                        ])):
                        
                        # Clean up title
                        title_text = re.sub(r'\s+', ' ', title_text)
                        title_text = re.sub(r'[^\w\s\-–&()[\]]', '', title_text)
                        
                        results.append({
                            "title": title_text,
                            "snippet": snippet_text[:200] if snippet_text else "",
                            "link": link_text,
                            "source": "browser_google"
                        })
                        
                        if len(results) >= 8:
                            break
                            
                except Exception as e:
                    logging.debug("Browser search: error extracting result %d: %s", i, e)
                    continue
            
            logging.debug("Browser search: successfully extracted %d results", len(results))
            
        finally:
            # Always close the browser
            try:
                driver.quit()
            except:
                pass
    
    except Exception as error:
        logging.debug("Browser search: failed: %s", error)
    
    return results


def google_search_filter_query_fallback(query: str) -> Optional[str]:
    """Fallback Google search filtering using web scraping.
    
    This function implements a layered fallback approach:
    1. LLM parsing (if enabled and available)
    2. Direct Google web search with BeautifulSoup parsing
    3. Query parsing fallback (always available as last resort)
    
    The function performs direct web scraping of Google search results
    to extract relevant information for query filtering.
    """
    config = get_runtime_config()
    if not config.filter_queries_with_google:
        return None

    # Try LLM parsing first if enabled
    if config.use_llm_google_parsing:
        llm_result = _google_search_filter_with_llm(query)
        if llm_result:
            return llm_result

    try:
        # Check if we should attempt web scraping fallback
        # This can be disabled to avoid unnecessary web requests
        if not getattr(config, 'use_simple_web_scraping_fallback', True):
            logging.debug("Google filter (fallback): web scraping fallback disabled")
            # Try query parsing as final fallback
            parsed_result = _parse_query_fallback(query)
            if parsed_result and isinstance(parsed_result, dict):
                if "query" in parsed_result and parsed_result["query"]:
                    logging.info(
                        "Google filter (fallback): query parsing fallback succeeded: '%s' -> '%s'",
                        query, 
                        parsed_result["query"]
                    )
                    return parsed_result["query"]
            return None
        
        if requests is None:
            logging.warning("Google filter (fallback): requests library not available")
            return None

        from bs4 import BeautifulSoup

        # Prepare search query - add music context for better results
        search_query = f'{query} song music'
        encoded_query = quote_plus(search_query)

        # Use a simple web search
        url = f"https://www.google.com/search?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        logging.debug(
            "Google filter (fallback): searching for '%s' with direct web scraping",
            search_query,
        )
        
        # Try multiple search approaches
        search_urls = [
            f"https://www.google.com/search?q={encoded_query}",
            f"https://www.google.com/search?q={encoded_query}&hl=en",
            f"https://www.google.com/search?q={encoded_query}&num=10",
        ]
        
        # Enhanced headers to avoid detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        
        search_results = None
        last_error = None
        
        for url in search_urls:
            try:
                logging.debug("Google filter (fallback): trying URL: %s", url)
                response = requests.get(url, headers=headers, timeout=15)
                
                # Check if we got a valid response
                if response.status_code != 200:
                    logging.debug("Google filter (fallback): HTTP %d for URL %s", response.status_code, url)
                    continue
                
                # Check if response contains expected content
                if not response.text or len(response.text) < 1000:
                    logging.debug("Google filter (fallback): response too short for URL %s", url)
                    continue
                
                # Check if we got blocked (common indicators)
                if any(indicator in response.text.lower() for indicator in [
                    "captcha", "robot", "automated", "blocked", "suspicious", 
                    "unusual traffic", "verify you are human", "security check"
                ]):
                    logging.debug("Google filter (fallback): detected blocking for URL %s", url)
                    continue
                
                # Check if we got actual search results
                # Google might return different content when blocking requests
                if not any(indicator in response.text.lower() for indicator in [
                    "search", "results", "google", "q=", "search?q="
                ]):
                    logging.debug("Google filter (fallback): no search results in response for URL %s", url)
                    continue
                
                # Check if we got a captcha or blocking page
                if any(blocked in response.text.lower() for blocked in [
                    "captcha", "robot", "automated", "blocked", "suspicious", 
                    "unusual traffic", "verify you are human", "security check",
                    "please complete the security check", "we're sorry"
                ]):
                    logging.debug("Google filter (fallback): detected blocking/captcha for URL %s", url)
                    continue
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract multiple search results and score them
                search_results = _extract_google_search_results_from_html(soup, query)
                
                if search_results:
                    logging.debug("Google filter (fallback): found %d results with URL %s", len(search_results), url)
                    break
                else:
                    logging.debug("Google filter (fallback): no results extracted from URL %s", url)
                    
            except Exception as error:
                last_error = error
                logging.debug("Google filter (fallback): error with URL %s: %s", url, error)
                continue
        
        if not search_results:
            logging.debug("Google filter (fallback): all search URLs failed, last error: %s", last_error)
            
            # Try browser-based search as fallback (most likely to succeed)
            if config.use_browser_based_search:
                try:
                    logging.debug("Google filter (fallback): trying browser-based search")
                    search_results = _try_browser_based_google_search(query)
                    if search_results:
                        logging.debug("Google filter (fallback): browser search found %d results", len(search_results))
                except Exception as browser_error:
                    logging.debug("Google filter (fallback): browser search failed: %s", browser_error)
            else:
                logging.debug("Google filter (fallback): browser-based search disabled in config")
            
            # If browser search failed, try alternative search engines
            if not search_results:
                try:
                    logging.debug("Google filter (fallback): trying alternative search engines")
                    search_results = _try_alternative_search_engines(query)
                except Exception as alt_error:
                    logging.debug("Google filter (fallback): alternative search engines also failed: %s", alt_error)
        
        # Add debugging information about what we found
        if search_results:
            logging.debug("Google filter (fallback): found %d total results", len(search_results))
            for i, result in enumerate(search_results[:3]):  # Log first 3 results
                logging.debug("Google filter (fallback): result %d: '%s' (source: %s)", 
                            i+1, result.get("title", "NO_TITLE"), result.get("source", "unknown"))
        else:
            logging.debug("Google filter (fallback): no results found from any source")
        
        if search_results:
            # Use the same scoring system as the API version
            best_result = _find_best_google_result(query, search_results)
            
            if best_result:
                logging.info(
                    "Google filter (fallback): '%s' -> '%s' (confidence=%.2f)",
                    query, best_result["title"], best_result["confidence"]
                )
                return best_result["title"]
            else:
                logging.debug("Google filter (fallback): scoring failed to find best result")
        else:
            logging.debug("Google filter (fallback): no search results to score")

        # Final fallback: try to parse the query itself for basic structure
        # This can help with queries that have clear separators like "Title - Artist"
        logging.debug("Google filter (fallback): trying query parsing fallback")
        parsed_result = _parse_query_fallback(query)
        if parsed_result and isinstance(parsed_result, dict):
            if "query" in parsed_result and parsed_result["query"]:
                logging.info(
                    "Google filter (fallback): query parsing fallback succeeded: '%s' -> '%s'",
                    query, 
                    parsed_result["query"]
                )
                return parsed_result["query"]
        
        return None

    except Exception as error:
        logging.debug(
            "Google filter (fallback): search failed for '%s': %s", query, error
        )
        
        # Even if the main search failed, try query parsing as a last resort
        try:
            logging.debug("Google filter (fallback): trying query parsing fallback after error")
            parsed_result = _parse_query_fallback(query)
            if parsed_result and isinstance(parsed_result, dict):
                if "query" in parsed_result and parsed_result["query"]:
                    logging.info(
                        "Google filter (fallback): query parsing fallback succeeded after error: '%s' -> '%s'",
                        query, 
                        parsed_result["query"]
                    )
                    return parsed_result["query"]
        except Exception as fallback_error:
            logging.debug("Google filter (fallback): query parsing fallback also failed: %s", fallback_error)
        
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

        # Prepare search query - add music context for better results
        search_query = f'{query} song music'
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
        html_content = soup.prettify()[:8000]  # Limit to first 8KB to avoid token limits

        # Prepare improved prompt for LLM
        prompt = f"""
You are analyzing a Google search results page for the query: "{query}"

Your task is to find the BEST music-related search result title that would be most useful for finding the actual song.

HTML content from the search results page:
{html_content}

Instructions:
1. Look for the FIRST actual search result (not ads, not navigation, not Google's own pages)
2. Focus on results that contain music-related content (songs, artists, albums, music videos)
3. Avoid results with words like "download", "free", "mp3", "buy", "sponsored", "advertisement"
4. Prefer results from platforms like YouTube, Spotify, Apple Music, or official artist websites
5. The result should be relevant to the original query "{query}"
6. Return ONLY the title text, nothing else
7. If no clear, relevant music result is found, return "NO_RESULT"

Examples of good results:
- "Bohemian Rhapsody - Queen (Official Video)"
- "Imagine - John Lennon (Lyrics)"
- "Hotel California - Eagles (Official Audio)"

Examples of bad results to avoid:
- "Download Free MP3 Songs"
- "Google Search Results"
- "Sponsored: Buy Music Online"

Return the title of the best music-related search result:
"""

        # Call LLM API
        llm_result = _call_llm_api(prompt, config.llm_api_key, config.llm_model)

        if llm_result and llm_result.strip() != "NO_RESULT":
            result_title = llm_result.strip()
            
            # Validate the LLM result using our scoring system
            if len(result_title) > 5:
                # Create a mock result object for scoring
                mock_result = {
                    "title": result_title,
                    "snippet": f"LLM extracted result for: {query}",
                    "link": ""
                }
                
                # Score the result to ensure it's not gibberish
                score = _score_google_result(query, result_title, "", "", 0)
                
                min_score = config.google_filter_llm_min_score
                if score >= min_score:
                    logging.info("Google filter (LLM): '%s' -> '%s' (score=%.1f)", 
                                query, result_title, score)
                    return result_title
                else:
                    logging.debug("Google filter (LLM): result scored too low (%.1f < %.1f): '%s'", 
                                score, min_score, result_title)

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


def test_google_search_debug(query: str = "test song") -> None:
    """Debug function to test Google search functionality."""
    import logging
    
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.DEBUG)
    
    print(f"Testing Google search with query: '{query}'")
    print("=" * 50)
    
    try:
        # Test the fallback method directly
        from .config import get_runtime_config
        config = get_runtime_config()
        
        print(f"Config: use_llm_google_parsing={config.use_llm_google_parsing}")
        print(f"Config: llm_api_key={'SET' if config.llm_api_key else 'NOT_SET'}")
        
        # Test the main function
        result = google_search_filter_query_main(query)
        
        if result:
            print(f"SUCCESS: Found result: '{result}'")
        else:
            print("FAILED: No result found")
            
    except Exception as error:
        print(f"ERROR: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with a sample query
    test_google_search_debug("Bohemian Rhapsody Queen")
