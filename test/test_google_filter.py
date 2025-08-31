#!/usr/bin/env python3
"""
Test script for Google query filtering
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yt_search_dl.config import RuntimeConfig, set_runtime_config
from yt_search_dl.google_search import google_search_filter_query_main

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def test_google_filter():
    """Test the Google filter functionality."""
    
    print("=== Google Query Filtering Test ===")
    print("This test shows why the filter returns None and how to fix it.\n")
    
    # Test without credentials (should return None)
    print("1. Testing without API credentials:")
    config = RuntimeConfig(filter_queries_with_google=True)
    set_runtime_config(config)
    
    result = google_search_filter_query_main("bohemian rhapsody")
    print(f"   Result: {result}")
    print("   → This is expected: API credentials are required\n")
    
    # Test with fallback enabled
    print("2. Testing with web scraping fallback:")
    config = RuntimeConfig(
        filter_queries_with_google=True,
        use_google_search_fallback=True
    )
    set_runtime_config(config)
    
    result = google_search_filter_query_main("bohemian rhapsody")
    print(f"   Result: {result}")
    print("   → Web scraping is unreliable due to Google's anti-bot measures\n")
    
    # Test with simple web scraping fallback disabled
    print("2a. Testing with simple web scraping fallback disabled:")
    config = RuntimeConfig(
        filter_queries_with_google=True,
        use_google_search_fallback=True,
        use_simple_web_scraping_fallback=False
    )
    set_runtime_config(config)
    
    result = google_search_filter_query_main("bohemian rhapsody")
    print(f"   Result: {result}")
    print("   → Only comprehensive search and query parsing are used\n")
    
    # Test with mock API credentials
    print("3. Testing with invalid API credentials:")
    config = RuntimeConfig(
        filter_queries_with_google=True,
        google_api_key="fake_key",
        google_search_engine_id="fake_engine"
    )
    set_runtime_config(config)
    
    result = google_search_filter_query_main("bohemian rhapsody")
    print(f"   Result: {result}")
    print("   → This fails because the credentials are invalid\n")
    
    print("=== SOLUTION ===")
    print("To use Google Query Filtering, you need:")
    print("1. Valid Google Custom Search API key")
    print("2. Valid Google Custom Search Engine ID")
    print("3. Run with: --filter-queries-with-google --google-api-key YOUR_KEY --google-search-engine-id YOUR_ID")
    print("\nExample:")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --google-api-key YOUR_ACTUAL_API_KEY \\")
    print("  --google-search-engine-id YOUR_ACTUAL_ENGINE_ID")
    
    print("\n=== FALLBACK OPTIONS ===")
    print("Control fallback behavior with additional options:")
    print("--use-google-search-fallback: Enable comprehensive web scraping fallback")
    print("--use-simple-web-scraping-fallback: Enable simple web scraping fallback (default: True)")
    print("--use-llm-google-parsing: Use LLM for parsing search results")
    print("\nExample with fallback control:")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --use-google-search-fallback \\")
    print("  --no-use-simple-web-scraping-fallback \\")
    print("  → Uses comprehensive search + query parsing only")
    
    print("\n=== LLM-BASED PARSING ===")
    print("For more reliable parsing without Google API:")
    print("1. Get an OpenAI API key")
    print("2. Install OpenAI: pip install openai")
    print("3. Run with: --filter-queries-with-google --use-llm-google-parsing --llm-api-key YOUR_OPENAI_KEY")
    print("\nExample:")
    print("python main.py --input songs.txt --filter-queries-with-google --use-llm-google-parsing \\")
    print("  --llm-api-key YOUR_OPENAI_API_KEY")
    
    print("\n=== LOCAL LLM PARSING ===")
    print("For privacy and cost savings, use local LLMs:")
    print("1. Install Ollama: https://ollama.ai/")
    print("2. Pull a model: ollama pull llama2")
    print("3. Run with: --filter-queries-with-google --use-llm-google-parsing --llm-base-url http://localhost:11434 --llm-model llama2")
    print("\nExample:")
    print("python main.py --input songs.txt --filter-queries-with-google --use-llm-google-parsing \\")
    print("  --llm-base-url http://localhost:11434 --llm-model llama2")

if __name__ == "__main__":
    test_google_filter()
