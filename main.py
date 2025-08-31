#!/usr/bin/env python3
"""
Main CLI entry point for YouTube Search to MP3 Downloader.

This is the command-line interface that uses the yt_search_dl package.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from yt_search_dl import (
    RuntimeConfig, DownloadResult, process_queries, process_urls,
    read_queries, configure_logging, set_runtime_config
)
from yt_search_dl.download import list_results_to_file
from yt_search_dl.google_search import google_search_filter_query_main


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search YouTube queries and download first result as MP3",
        epilog="""
Examples:
  # Basic usage
  python main.py --input songs.txt --output downloads
  
  # Deep search for better matches
  python main.py --input songs.txt --deep-search --search-count 25
  
  # Search with title only (removes artist names)
  python main.py --input songs.txt --search-without-authors
  
  # Filter queries through Google search first
  python main.py --input songs.txt --filter-queries-with-google --google-api-key YOUR_KEY --google-search-engine-id YOUR_ID
  
  # Combine multiple features for maximum coverage
  python main.py --input songs.txt --deep-search --search-without-authors --filter-queries-with-google
        """
    )
    parser.add_argument("--input", required=True, help="Path to queries file (one query per line)")
    parser.add_argument("--output", default="downloads", help="Output directory for MP3 files")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between queries in seconds")
    parser.add_argument(
        "--search-count",
        type=int,
        default=20,
        help="Number of search results to consider (1-50).",
    )
    parser.add_argument(
        "--select-strategy",
        default="best",
        choices=["best", "first"],
        help="How to pick the result: best (exact/closest title) or first.",
    )
    parser.add_argument(
        "--use-ai-match",
        action="store_true",
        help="Enable AI semantic matching (sentence-transformers) for ranking",
    )
    parser.add_argument(
        "--ai-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name to use when --use-ai-match",
    )
    parser.add_argument(
        "--debug-matching",
        action="store_true",
        help="Log top candidates with score breakdown for matching",
    )
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=10,
        help="How many top candidates to show when --debug-matching",
    )
    parser.add_argument(
        "--rate-limit-kbps",
        type=int,
        default=None,
        help="Limit download speed in KiB/s (e.g., 512). Omit to disable",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of queries to process in parallel (IO-bound). Default 1",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    parser.add_argument(
        "--log-file",
        default="logs/run.log",
        help="Path to write detailed logs",
    )
    parser.add_argument(
        "--csv-logging",
        action="store_true",
        help="Enable CSV logging for concurrent operations (creates separate CSV file)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List search result links to a file instead of downloading",
    )
    parser.add_argument(
        "--list-output",
        default="links.txt",
        help="Path to write links when using --list-only",
    )
    parser.add_argument(
        "--from-links",
        help="Path to a file containing one URL per line to download as MP3",
    )
    parser.add_argument(
        "--use-spotify",
        action="store_true",
        help="Use Spotify to enrich queries (Client Credentials flow)",
    )
    parser.add_argument(
        "--spotify-client-id",
        help="Spotify Client ID (required when --use-spotify)",
    )
    parser.add_argument(
        "--spotify-client-secret",
        help="Spotify Client Secret (required when --use-spotify)",
    )
    parser.add_argument(
        "--spotify-limit",
        type=int,
        default=5,
        help="Max Spotify track results to consider (default: 5)",
    )
    parser.add_argument(
        "--spotify-min-score",
        type=float,
        default=0.35,
        help="Minimum Spotify match score to accept enrichment (0..1, default: 0.35)",
    )
    parser.add_argument(
        "--deep-search",
        action="store_true",
        help="Enable deeper search by doubling the number of results considered (up to 50)",
    )
    parser.add_argument(
        "--search-without-authors",
        action="store_true",
        help="Remove author/artist names from queries and search with just the title",
    )
    parser.add_argument(
        "--audio-quality",
        default="192",
        help="Audio quality in kbps (e.g., 192, 320) or 'best' for highest available",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "m4a", "opus", "flac"],
        help="Audio format for output files",
    )
    parser.add_argument(
        "--cookies-file",
        help="Path to cookies.txt file for YouTube authentication (handles age restrictions)",
    )
    parser.add_argument(
        "--use-google-search",
        action="store_true",
        help="Use Google search to enrich queries with additional context",
    )
    parser.add_argument(
        "--google-api-key",
        help="Google Custom Search API key (required when --use-google-search)",
    )
    parser.add_argument(
        "--google-search-engine-id",
        help="Google Custom Search Engine ID (required when --use-google-search)",
    )
    parser.add_argument(
        "--google-min-confidence",
        type=float,
        default=0.3,
        help="Minimum confidence score for Google search enrichment (0.0-1.0, default: 0.3)",
    )
    parser.add_argument(
        "--use-google-search-fallback",
        action="store_true",
        help="Use web scraping fallback for Google search (when API is not available)",
    )
    parser.add_argument(
        "--filter-queries-with-google",
        action="store_true",
        help="Filter queries through Google search first and use first result details as input",
    )
    parser.add_argument(
        "--use-simple-web-scraping-fallback",
        action="store_true",
        default=True,
        help="Enable simple web scraping fallback when comprehensive Google search fails (default: True)",
    )
    parser.add_argument(
        "--use-browser-based-search",
        action="store_true",
        default=True,
        help="Enable browser-based Google search using Selenium to bypass restrictions (default: True)",
    )
    parser.add_argument(
        "--use-llm-google-parsing",
        action="store_true",
        help="Use LLM to parse Google search results (requires --llm-api-key)",
    )
    parser.add_argument(
        "--llm-api-key",
        help="API key for LLM service (OpenAI, Anthropic, etc.) for parsing Google results",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-3.5-turbo",
        help="LLM model to use for parsing (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Base URL for local LLM service (e.g., http://localhost:11434 for Ollama)",
    )
    parser.add_argument(
        "--google-filter-min-score",
        type=float,
        default=30.0,
        help="Minimum score for Google search filtering API results (default: 30.0)",
    )
    parser.add_argument(
        "--google-filter-llm-min-score",
        type=float,
        default=20.0,
        help="Minimum score for Google search filtering LLM results (default: 20.0)",
    )
    parser.add_argument(
        "--no-google-filter-boost-music",
        action="store_true",
        help="Disable boosting of music-related content in Google filtering",
    )
    parser.add_argument(
        "--no-google-filter-penalize-spam",
        action="store_true",
        help="Disable penalizing of spam/ad content in Google filtering",
    )
    parser.add_argument(
        "--no-google-filter-prefer-video",
        action="store_true",
        help="Disable preference for video platform results in Google filtering",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()
    list_output = Path(args.list_output).expanduser().resolve() if getattr(args, "list_output", None) else None

    configure_logging(args.log_level, log_file, args.csv_logging)
    logging.info("Input file: %s", input_path)
    logging.info("Output dir: %s", output_dir)
    logging.debug("Log file: %s", log_file)
    
    if args.csv_logging:
        from yt_search_dl.utils import get_csv_log_file
        csv_file = get_csv_log_file()
        if csv_file:
            logging.info("CSV logging enabled: %s", csv_file)
    
    # Log feature flags
    if args.deep_search:
        logging.info("Deep search mode enabled: will consider up to 50 results")
    if args.search_without_authors:
        logging.info("Title-only search enabled: will remove author names from queries")
    if args.audio_quality != "192" or args.audio_format != "mp3":
        logging.info("Audio settings: quality=%s, format=%s", args.audio_quality, args.audio_format)
    if getattr(args, "cookies_file", None):
        logging.info("Using cookies file for authentication: %s", args.cookies_file)
    if args.use_google_search:
        logging.info("Google search enrichment enabled")
        if args.use_google_search_fallback:
            logging.info("Google search fallback (web scraping) enabled")
    if args.filter_queries_with_google:
        logging.info("Google query filtering enabled: will filter queries through Google search first")
        if not args.use_simple_web_scraping_fallback:
            logging.info("Simple web scraping fallback disabled: will only use comprehensive search and query parsing")
        if args.use_browser_based_search:
            logging.info("Browser-based Google search enabled: will use Selenium to bypass restrictions")
        if args.use_llm_google_parsing:
            logging.info("LLM Google parsing enabled: will use LLM to parse search results")

    try:
        queries = read_queries(input_path)
        if not queries:
            logging.warning("No queries found in %s", input_path)
            return 0
    except Exception as error: # noqa: BLE001
        logging.error("Failed to read queries: %s", error)
        return 1

    # Set runtime configuration first (needed for Google search filtering)
    config = RuntimeConfig(
        search_count=int(args.search_count),
        select_strategy=str(args.select_strategy),
        use_ai_match=bool(args.use_ai_match),
        ai_model_name=str(args.ai_model),
        use_spotify=bool(args.use_spotify),
        spotify_client_id=str(args.spotify_client_id) if getattr(args, "spotify_client_id", None) else None,
        spotify_client_secret=str(args.spotify_client_secret) if getattr(args, "spotify_client_secret", None) else None,
        spotify_limit=int(args.spotify_limit),
        spotify_min_score=float(args.spotify_min_score),
        debug_matching=bool(args.debug_matching),
        debug_top_k=int(args.debug_topk),
        deep_search=bool(args.deep_search),
        search_without_authors=bool(args.search_without_authors),
        audio_quality=str(args.audio_quality),
        audio_format=str(args.audio_format),
        cookies_file=str(args.cookies_file) if getattr(args, "cookies_file", None) else None,
        use_google_search=bool(args.use_google_search),
        google_api_key=str(args.google_api_key) if getattr(args, "google_api_key", None) else None,
        google_search_engine_id=str(args.google_search_engine_id) if getattr(args, "google_search_engine_id", None) else None,
        google_min_confidence=float(args.google_min_confidence),
        use_google_search_fallback=bool(args.use_google_search_fallback),
        filter_queries_with_google=bool(args.filter_queries_with_google),
        use_simple_web_scraping_fallback=bool(args.use_simple_web_scraping_fallback),
        use_browser_based_search=bool(args.use_browser_based_search),
        use_llm_google_parsing=bool(args.use_llm_google_parsing),
        llm_api_key=str(args.llm_api_key) if getattr(args, "llm_api_key", None) else None,
        llm_model=str(args.llm_model),
        llm_base_url=str(args.llm_base_url) if getattr(args, "llm_base_url", None) else None,
        google_filter_min_score=float(args.google_filter_min_score),
        google_filter_llm_min_score=float(args.google_filter_llm_min_score),
        google_filter_boost_music_keywords=not bool(args.no_google_filter_boost_music),
        google_filter_penalize_spam=not bool(args.no_google_filter_penalize_spam),
        google_filter_prefer_video_platforms=not bool(args.no_google_filter_prefer_video),
    )
    set_runtime_config(config)

    # Apply Google search filtering if enabled
    if args.filter_queries_with_google:
        logging.info("Applying Google search filtering to %d queries...", len(queries))
        filtered_queries = []
        for i, query in enumerate(queries, 1):
            logging.info("[%d/%d] Filtering query: %s", i, len(queries), query)
            filtered_query = google_search_filter_query_main(query)
            if filtered_query:
                filtered_queries.append(filtered_query)
                logging.info("[%d/%d] Filtered: '%s' -> '%s'", i, len(queries), query, filtered_query)
            else:
                filtered_queries.append(query)
                logging.info("[%d/%d] No filter result, using original: %s", i, len(queries), query)
        queries = filtered_queries

    # Listing mode: write links and exit
    if getattr(args, "list_only", False):
        logging.info("Listing results only. Output file: %s", list_output)
        list_results_to_file(
            queries=queries,
            output_file=list_output or Path("links.txt").resolve(),
            search_count=int(args.search_count),
            delay_seconds=float(args.delay),
        )
        logging.info("Done writing links to %s", list_output)
        return 0

    # Download from a file of URLs (one per line)
    if getattr(args, "from_links", None):
        links_path = Path(args.from_links).expanduser().resolve()
        try:
            raw = links_path.read_text(encoding="utf-8").splitlines()
        except Exception as error:  # noqa: BLE001
            logging.error("Failed to read links file %s: %s", links_path, error)
            return 1

        # Filter out comments and blanks
        urls = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]
        if not urls:
            logging.warning("No URLs found in %s", links_path)
            return 0

        results = process_urls(
            urls=urls,
            output_dir=output_dir,
            delay_seconds=float(args.delay),
            rate_limit_kbps=args.rate_limit_kbps,
            concurrency=int(args.concurrency),
        )
        total = len(results)
        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded
        logging.info("Completed (from links): %d succeeded, %d failed", succeeded, failed)
        return 0 if failed == 0 else 2

    results = process_queries(
        queries=queries,
        output_dir=output_dir,
        delay_seconds=float(args.delay),
        rate_limit_kbps=args.rate_limit_kbps,
        concurrency=int(args.concurrency),
    )

    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded
    logging.info("Completed: %d succeeded, %d failed", succeeded, failed)

    # Non-zero exit if any failed
    return 0 if failed == 0 else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
