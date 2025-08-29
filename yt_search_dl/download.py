"""
Download and processing functionality for YouTube Search to MP3 Downloader.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List

# yt-dlp will be imported when needed in functions

from .config import DownloadResult, get_runtime_config
from .search import search_video_url, collect_search_urls
from .utils import ensure_ffmpeg_available


def build_ydl_opts(output_dir: Path, rate_limit_kbps: int | None) -> dict:
    """Create yt-dlp options for MP3 extraction and logging."""
    # Template: Title.mp3 inside output directory
    outtmpl = str(output_dir / "%(title)s.%(ext)s")
    ydl_opts: dict = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": outtmpl,
        "format": "bestaudio/best",
        "nocheckcertificate": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        # Write metadata & thumbnail into the file when possible
        "writethumbnail": True,
        "embedthumbnail": True,
        "addmetadata": True,
        "postprocessor_args": [
            "-metadata",
            "comment=Downloaded via yt-dlp",
        ],
        # Retry policy
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 20,
    }

    if rate_limit_kbps and rate_limit_kbps > 0:
        # yt-dlp expects bytes per second
        ydl_opts["ratelimit"] = int(rate_limit_kbps * 1024)

    return ydl_opts


def download_mp3_from_url(url: str, ydl_opts: dict) -> None:
    """Download and convert the given YouTube URL to mp3 using yt-dlp."""
    import yt_dlp  # type: ignore
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def list_results_to_file(
    queries: Iterable[str],
    output_file: Path,
    search_count: int,
    delay_seconds: float,
) -> None:
    """Write all found result links per query into a file. Does not download."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Stream write to file as we go, so results appear immediately
    with output_file.open("w", encoding="utf-8") as fh:
        for idx, query in enumerate(queries, start=1):
            logging.info("[%d] Listing results for: %s", idx, query)
            urls = collect_search_urls(query, search_count)
            fh.write(f"# query: {query}\n")
            if urls:
                for url in urls:
                    fh.write(f"{url}\n")
            else:
                fh.write("# no results\n")
            fh.write("\n")  # blank line between queries
            fh.flush()
            time.sleep(delay_seconds)


def process_queries(
    queries: Iterable[str],
    output_dir: Path,
    delay_seconds: float,
    rate_limit_kbps: int | None,
    concurrency: int = 1,
) -> List[DownloadResult]:
    """Process queries and download results."""
    ensure_ffmpeg_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = build_ydl_opts(output_dir, rate_limit_kbps)
    config = get_runtime_config()

    # Worker function for one query
    def process_one(args_tuple: tuple[int, str]) -> DownloadResult:
        idx, query = args_tuple
        logging.info("[%d] Searching: %s", idx, query)
        url = search_video_url(query, config.search_count, config.select_strategy)
        if not url:
            msg = "No results found"
            logging.warning("[%d] %s: %s", idx, msg, query)
            time.sleep(delay_seconds)
            return DownloadResult(query=query, url=None, success=False, reason=msg)

        logging.info("[%d] Found: %s", idx, url)
        try:
            # Use per-call options to avoid accidental mutation
            download_mp3_from_url(url, dict(ydl_opts))
            logging.info("[%d] Downloaded OK: %s", idx, url)
            return DownloadResult(query=query, url=url, success=True)
        except Exception as error:  # noqa: BLE001
            logging.error("[%d] Download failed: %s | %s", idx, url, error)
            return DownloadResult(query=query, url=url, success=False, reason=str(error))
        finally:
            # Space out requests per worker
            time.sleep(delay_seconds)

    # Run sequentially if concurrency <= 1 for backward compatibility
    if concurrency <= 1:
        results: List[DownloadResult] = []
        for pair in enumerate(queries, start=1):
            results.append(process_one(pair))
        return results

    # Concurrency path for IO-bound search+download
    from concurrent.futures import ThreadPoolExecutor

    indexed_queries = list(enumerate(queries, start=1))
    max_workers = max(1, int(concurrency))
    results_parallel: List[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="yt-dlp") as executor:
        for result in executor.map(process_one, indexed_queries):
            results_parallel.append(result)

    return results_parallel


def process_urls(
    urls: Iterable[str],
    output_dir: Path,
    delay_seconds: float,
    rate_limit_kbps: int | None,
    concurrency: int = 1,
) -> List[DownloadResult]:
    """Download a list of direct video URLs as MP3 using yt-dlp."""
    ensure_ffmpeg_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = build_ydl_opts(output_dir, rate_limit_kbps)

    def process_one(args_tuple: tuple[int, str]) -> DownloadResult:
        idx, url = args_tuple
        logging.info("[%d] Downloading: %s", idx, url)
        try:
            download_mp3_from_url(url, dict(ydl_opts))
            logging.info("[%d] Downloaded OK: %s", idx, url)
            return DownloadResult(query=url, url=url, success=True)
        except Exception as error:  # noqa: BLE001
            logging.error("[%d] Download failed: %s | %s", idx, url, error)
            return DownloadResult(query=url, url=url, success=False, reason=str(error))
        finally:
            time.sleep(delay_seconds)

    indexed_urls = [(i, u) for i, u in enumerate(urls, start=1)]
    if concurrency <= 1:
        results: List[DownloadResult] = []
        for pair in indexed_urls:
            results.append(process_one(pair))
        return results

    from concurrent.futures import ThreadPoolExecutor

    results_parallel: List[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency)), thread_name_prefix="yt-dlp") as executor:
        for result in executor.map(process_one, indexed_urls):
            results_parallel.append(result)
    return results_parallel


__all__ = [
    "build_ydl_opts",
    "download_mp3_from_url", 
    "list_results_to_file",
    "process_queries",
    "process_urls",
]
