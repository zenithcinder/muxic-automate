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
  
  # Combine both features for maximum coverage
  python main.py --input songs.txt --deep-search --search-without-authors
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
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    log_file = Path(args.log_file).expanduser().resolve()
    list_output = Path(args.list_output).expanduser().resolve() if getattr(args, "list_output", None) else None

    configure_logging(args.log_level, log_file)
    logging.info("Input file: %s", input_path)
    logging.info("Output dir: %s", output_dir)
    logging.debug("Log file: %s", log_file)
    
    # Log feature flags
    if args.deep_search:
        logging.info("Deep search mode enabled: will consider up to 50 results")
    if args.search_without_authors:
        logging.info("Title-only search enabled: will remove author names from queries")

    try:
        queries = read_queries(input_path)
        if not queries:
            logging.warning("No queries found in %s", input_path)
            return 0
    except Exception as error:  # noqa: BLE001
        logging.error("Failed to read queries: %s", error)
        return 1

    # Set runtime configuration
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
    )
    set_runtime_config(config)

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
