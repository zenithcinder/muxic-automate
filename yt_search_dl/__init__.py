"""
YouTube Search to MP3 Downloader Package

A batch YouTube search and MP3 downloader with advanced matching capabilities.
"""

__version__ = "1.0.0"
__author__ = "YouTube Search to MP3 Team"

from .config import RuntimeConfig, DownloadResult, set_runtime_config
from .search import search_video_url, collect_search_urls
from .download import process_queries, process_urls
from .utils import read_queries, configure_logging

__all__ = [
    "RuntimeConfig",
    "DownloadResult", 
    "set_runtime_config",
    "search_video_url",
    "collect_search_urls",
    "process_queries",
    "process_urls",
    "read_queries",
    "configure_logging",
]
