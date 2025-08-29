"""
Spotify integration for YouTube Search to MP3 Downloader.
"""

import base64
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .config import get_runtime_config, get_spotify_token, set_spotify_token
from .utils import _normalize_text, _similarity, _tokenize, _token_jaccard


def _get_spotify_token(client_id: str, client_secret: str) -> str | None:
    """Get Spotify access token using client credentials flow."""
    now = int(time.time())
    token = get_spotify_token()
    if token and token.get("expires_at", 0) - 30 > now:
        return token.get("access_token")

    try:
        creds = f"{client_id}:{client_secret}".encode("utf-8")
        auth = base64.b64encode(creds).decode("ascii")
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("ascii")
        req = urllib.request.Request(
            "https://accounts.spotify.com/api/token",
            data=data,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        access_token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        token_data = {"access_token": access_token, "expires_at": now + expires_in}
        set_spotify_token(token_data)
        logging.info("Spotify: obtained access token (expires in %ss)", expires_in)
        return access_token
    except Exception as error:  # noqa: BLE001
        logging.error("Spotify: failed to obtain token: %s", error)
        return None


def _spotify_search_first_track(raw_query: str) -> dict | None:
    """Return best-matching Spotify track {title, artists, duration_sec} for the query.

    Searches up to spotify_limit results and ranks against the raw query using title/author
    similarity and duration preference (<= 10 minutes). Only returns a result if score
    exceeds spotify_min_score; otherwise returns None to avoid harming relevance.
    """
    config = get_runtime_config()
    if not config.use_spotify:
        return None
    client_id = config.spotify_client_id
    client_secret = config.spotify_client_secret
    if not client_id or not client_secret:
        logging.warning("Spotify: --use-spotify set but client credentials are missing")
        return None

    token = _get_spotify_token(client_id, client_secret)
    if not token:
        return None
    limit = max(1, min(config.spotify_limit, 50))
    try:
        q = urllib.parse.urlencode({"q": raw_query, "type": "track", "limit": limit})
        url = f"https://api.spotify.com/v1/search?{q}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = (((payload or {}).get("tracks") or {}).get("items") or [])
        if not items:
            logging.info("Spotify: no track found for '%s'", raw_query)
            return None

        # Rank tracks by similarity to the raw user query
        query_norm = _normalize_text(raw_query)
        query_tokens = _tokenize(raw_query)
        def sp_score(track: dict) -> float:
            name = track.get("name") or ""
            artists = [a.get("name") for a in (track.get("artists") or []) if isinstance(a.get("name"), str)]
            authors_text = " ".join(artists)
            title_norm = _normalize_text(name)
            s_title = _similarity(title_norm, query_norm)
            s_author = _similarity(_normalize_text(authors_text), query_norm)
            s_jacc = _token_jaccard(_tokenize(name), query_tokens)
            # Duration preference: punish >10min and very long mismatches
            duration_ms = int(track.get("duration_ms") or 0)
            duration_sec = duration_ms // 1000 if duration_ms else None
            dur_penalty = 0.0
            if duration_sec and duration_sec > 600:
                dur_penalty = -0.2
            return 0.5 * s_title + 0.3 * s_author + 0.2 * s_jacc + dur_penalty

        ranked = sorted(items, key=sp_score, reverse=True)
        top = ranked[0]
        top_score = sp_score(top)
        threshold = config.spotify_min_score
        if top_score < threshold:
            logging.info("Spotify: top match below threshold (%.2f < %.2f); skipping enrichment", top_score, threshold)
            return None

        name = top.get("name") or ""
        artists = [a.get("name") for a in (top.get("artists") or []) if isinstance(a.get("name"), str)]
        duration_ms = int(top.get("duration_ms") or 0)
        duration_sec = duration_ms // 1000 if duration_ms else None
        logging.info("Spotify: selected '%s' by %s (score=%.2f, %ss)", name, ", ".join(artists), top_score, duration_sec or 0)
        return {"title": name, "artists": artists, "duration_sec": duration_sec}
    except Exception as error:  # noqa: BLE001
        logging.error("Spotify: search failed for '%s': %s", raw_query, error)
        return None


__all__ = ["_spotify_search_first_track"]
