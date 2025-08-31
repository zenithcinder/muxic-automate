#!/usr/bin/env python3
"""
Test script for improved Google search filtering

This script demonstrates the enhanced filtering that prevents gibberish results
by using intelligent scoring and validation.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yt_search_dl.config import RuntimeConfig, set_runtime_config
from yt_search_dl.google_search import google_search_filter_query_main

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_improved_filter():
    """Test the improved Google filter functionality."""
    
    print("=== Improved Google Search Filtering Test ===")
    print("This test demonstrates the enhanced filtering that prevents gibberish results.\n")
    
    # Test with different configuration settings
    test_configs = [
        {
            "name": "Default settings (balanced)",
            "config": RuntimeConfig(
                filter_queries_with_google=True,
                google_filter_min_score=30.0,
                google_filter_llm_min_score=20.0,
                google_filter_boost_music_keywords=True,
                google_filter_penalize_spam=True,
                google_filter_prefer_video_platforms=True
            )
        },
        {
            "name": "Strict filtering (high quality only)",
            "config": RuntimeConfig(
                filter_queries_with_google=True,
                google_filter_min_score=50.0,
                google_filter_llm_min_score=30.0,
                google_filter_boost_music_keywords=True,
                google_filter_penalize_spam=True,
                google_filter_prefer_video_platforms=True
            )
        },
        {
            "name": "Lenient filtering (more results)",
            "config": RuntimeConfig(
                filter_queries_with_google=True,
                google_filter_min_score=20.0,
                google_filter_llm_min_score=15.0,
                google_filter_boost_music_keywords=True,
                google_filter_penalize_spam=True,
                google_filter_prefer_video_platforms=True
            )
        },
        {
            "name": "No music boost (neutral scoring)",
            "config": RuntimeConfig(
                filter_queries_with_google=True,
                google_filter_min_score=30.0,
                google_filter_llm_min_score=20.0,
                google_filter_boost_music_keywords=False,
                google_filter_penalize_spam=True,
                google_filter_prefer_video_platforms=True
            )
        },
        {
            "name": "No spam penalty (allows more content)",
            "config": RuntimeConfig(
                filter_queries_with_google=True,
                google_filter_min_score=30.0,
                google_filter_llm_min_score=20.0,
                google_filter_boost_music_keywords=True,
                google_filter_penalize_spam=False,
                google_filter_prefer_video_platforms=True
            )
        }
    ]
    
    test_queries = [
        "bohemian rhapsody queen",
        "imagine john lennon",
        "hotel california eagles",
        "stairway to heaven led zeppelin",
        "hey jude beatles"
    ]
    
    for test_config in test_configs:
        print(f"\n--- {test_config['name']} ---")
        set_runtime_config(test_config['config'])
        
        for query in test_queries[:2]:  # Test first 2 queries for each config
            print(f"  Query: '{query}'")
            
            # Note: This will return None without actual API credentials
            # but demonstrates the configuration system
            result = google_search_filter_query_main(query)
            
            if result:
                print(f"    → Filtered: '{result}'")
            else:
                print(f"    → No filter result (expected without API credentials)")
        
        print(f"  Min score threshold: {test_config['config'].google_filter_min_score}")
        print(f"  Music boost: {test_config['config'].google_filter_boost_music_keywords}")
        print(f"  Spam penalty: {test_config['config'].google_filter_penalize_spam}")
        print(f"  Video preference: {test_config['config'].google_filter_prefer_video_platforms}")
    
    print("\n=== Key Improvements ===")
    print("1. Intelligent scoring system that evaluates result quality")
    print("2. Configurable thresholds for different filtering strictness")
    print("3. Music-specific content boosting")
    print("4. Spam and advertisement filtering")
    print("5. Video platform preference for music content")
    print("6. Word overlap analysis with original query")
    print("7. Length and formatting validation")
    print("8. Position-based scoring (first results get higher scores)")
    
    print("\n=== Usage Examples ===")
    print("# Default balanced filtering:")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --google-api-key YOUR_KEY --google-search-engine-id YOUR_ID")
    
    print("\n# Strict high-quality filtering:")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --google-filter-min-score 50.0 --google-filter-llm-min-score 30.0")
    
    print("\n# Lenient filtering (more results):")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --google-filter-min-score 20.0 --google-filter-llm-min-score 15.0")
    
    print("\n# Customize filtering behavior:")
    print("python main.py --input songs.txt --filter-queries-with-google \\")
    print("  --no-google-filter-boost-music --no-google-filter-penalize-spam")

if __name__ == "__main__":
    test_improved_filter()
