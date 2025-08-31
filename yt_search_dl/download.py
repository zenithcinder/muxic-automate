"""
Download and processing functionality for YouTube Search to MP3 Downloader.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Iterable, List
from .utils import log_download_result_csv

# yt-dlp will be imported when needed in functions

from .config import DownloadResult, get_runtime_config
from .search import search_video_url, collect_search_urls
from .utils import ensure_ffmpeg_available


def build_ydl_opts(output_dir: Path, rate_limit_kbps: int | None, audio_quality: str = "192", audio_format: str = "mp3", cookies_file: str | None = None) -> dict:
    """Create yt-dlp options for audio extraction and logging.
    
    Args:
        output_dir: Directory to save files
        rate_limit_kbps: Download speed limit in kbps
        audio_quality: Audio quality in kbps (e.g., "192", "320", "best")
        audio_format: Audio format ("mp3", "m4a", "opus", "flac")
        cookies_file: Path to cookies.txt file for authentication
    """
    # Template: Title.ext inside output directory
    outtmpl = str(output_dir / "%(title)s.%(ext)s")
    
    # Configure format selection based on desired quality
    if audio_quality == "best":
        # Get the best available audio quality
        format_spec = "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best"
    else:
        # Try to get specific quality, fallback to best
        format_spec = f"bestaudio[height<={audio_quality}]/bestaudio/best"
    
    ydl_opts: dict = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": outtmpl,
        "format": format_spec,
        "nocheckcertificate": True,
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
    
    # Add cookies file for authentication (handles age restrictions)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
        logging.debug("Using cookies file for authentication: %s", cookies_file)
    
    # Configure postprocessors based on desired format
    if audio_format.lower() == "mp3":
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": audio_quality,
            }
        ]
    elif audio_format.lower() == "m4a":
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": audio_quality,
            }
        ]
    elif audio_format.lower() == "opus":
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "opus",
                "preferredquality": audio_quality,
            }
        ]
    elif audio_format.lower() == "flac":
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "flac",
            }
        ]
    else:
        # Default to MP3
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": audio_quality,
            }
        ]

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
    config = get_runtime_config()
    ydl_opts = build_ydl_opts(output_dir, rate_limit_kbps, config.audio_quality, config.audio_format, config.cookies_file)

    # Worker function for one query
    def process_one(args_tuple: tuple[int, str]) -> DownloadResult:
        idx, query = args_tuple
        start_time = time.time()
        
        # Log start of processing
        log_download_result_csv(query, None, False, None, idx, None, "started")
        logging.info("[%d] Searching: %s", idx, query)
        
        url = search_video_url(query, config.search_count, config.select_strategy)
        if not url:
            msg = "No results found"
            duration_ms = (time.time() - start_time) * 1000
            logging.warning("[%d] %s: %s", idx, msg, query)
            log_download_result_csv(query, None, False, msg, idx, duration_ms, "failed")
            time.sleep(delay_seconds)
            return DownloadResult(query=query, url=None, success=False, reason=msg)

        logging.info("[%d] Found: %s", idx, url)
        try:
            # Use per-call options to avoid accidental mutation
            download_mp3_from_url(url, dict(ydl_opts))
            duration_ms = (time.time() - start_time) * 1000
            logging.info("[%d] Downloaded OK: %s", idx, url)
            log_download_result_csv(query, url, True, None, idx, duration_ms, "completed")
            return DownloadResult(query=query, url=url, success=True)
        except Exception as error:  # noqa: BLE001
            duration_ms = (time.time() - start_time) * 1000
            logging.error("[%d] Download failed: %s | %s", idx, url, error)
            log_download_result_csv(query, url, False, str(error), idx, duration_ms, "failed")
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
    """Download a list of direct video URLs as audio using yt-dlp."""
    ensure_ffmpeg_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = get_runtime_config()
    ydl_opts = build_ydl_opts(output_dir, rate_limit_kbps, config.audio_quality, config.audio_format, config.cookies_file)

    def process_one(args_tuple: tuple[int, str]) -> DownloadResult:
        idx, url = args_tuple
        start_time = time.time()
        
        # Log start of processing
        log_download_result_csv(url, None, False, None, idx, None, "started")
        logging.info("[%d] Downloading: %s", idx, url)
        
        try:
            download_mp3_from_url(url, dict(ydl_opts))
            duration_ms = (time.time() - start_time) * 1000
            logging.info("[%d] Downloaded OK: %s", idx, url)
            log_download_result_csv(url, url, True, None, idx, duration_ms, "completed")
            return DownloadResult(query=url, url=url, success=True)
        except Exception as error:  # noqa: BLE001
            duration_ms = (time.time() - start_time) * 1000
            logging.error("[%d] Download failed: %s | %s", idx, url, error)
            log_download_result_csv(url, url, False, str(error), idx, duration_ms, "failed")
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
