#!/usr/bin/env python3
"""Test script for browser-based Google search functionality."""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_browser_search():
    """Test the browser-based Google search functionality."""
    try:
        from yt_search_dl.google_search import _try_browser_based_google_search
        
        print("Testing browser-based Google search...")
        print("=" * 60)
        
        # Test query
        query = "Bohemian Rhapsody Queen"
        print(f"Query: '{query}'")
        
        # Test the browser search function
        results = _try_browser_based_google_search(query)
        
        if results:
            print(f"✅ SUCCESS: Found {len(results)} results")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"  Result {i+1}: '{result.get('title', 'NO_TITLE')}'")
                print(f"    Source: {result.get('source', 'unknown')}")
                print(f"    Snippet: {result.get('snippet', '')[:100]}...")
                print()
        else:
            print("❌ FAILED: No results found")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_browser_search()
