#!/usr/bin/env python3
"""Debug script to see the actual HTML structure from browser-based Google search."""

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

def debug_browser_html():
    """Debug the actual HTML structure from browser-based Google search."""
    try:
        from selenium import webdriver  # type: ignore[import-untyped]
        from selenium.webdriver.chrome.options import Options  # type: ignore[import-untyped]
        from selenium.webdriver.common.by import By  # type: ignore[import-untyped]
        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore[import-untyped]
        from selenium.webdriver.support import expected_conditions as EC  # type: ignore[import-untyped]
        
        print("Starting browser debug...")
        print("=" * 60)
        
        # Set up Chrome options
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = webdriver.Chrome(options=options)
        
        try:
            # Navigate to Google search
            query = "Bohemian Rhapsody Queen song music"
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            print(f"Navigating to: {url}")
            
            driver.get(url)
            
            # Wait a bit for page to load
            import time
            time.sleep(3)
            
            # Get page source
            page_source = driver.page_source
            
            print(f"\nPage source length: {len(page_source)}")
            print("\nFirst 2000 characters of page source:")
            print("-" * 60)
            print(page_source[:2000])
            print("-" * 60)
            
            # Try to find any elements that might contain search results
            print("\nTrying to find search result elements...")
            
            # Test various selectors
            selectors_to_test = [
                "div[data-sokoban-container]",
                "div.g",
                "div[jscontroller]",
                "div[data-ved]",
                "div[class*='g']",
                "div[class*='result']",
                "div[class*='tF2Cxc']",
                "div[class*='yuRUbf']",
                "div[class*='rc']",
                "div[class*='bkWMgd']",
                "div[class*='search']",
                "div[class*='main']",
                "div[class*='content']",
            ]
            
            for selector in selectors_to_test:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        print(f"✅ Selector '{selector}': found {len(elements)} elements")
                        # Show text from first element
                        if elements[0].text.strip():
                            print(f"   First element text: {elements[0].text.strip()[:100]}...")
                        else:
                            print(f"   First element has no text")
                    else:
                        print(f"❌ Selector '{selector}': no elements found")
                except Exception as e:
                    print(f"❌ Selector '{selector}': error - {e}")
            
            # Look for any text that might be search results
            print("\nLooking for any text that might contain search results...")
            all_text = driver.find_element(By.TAG_NAME, "body").text
            print(f"Total page text length: {len(all_text)}")
            print("First 500 characters of page text:")
            print("-" * 40)
            print(all_text[:500])
            print("-" * 40)
            
        finally:
            driver.quit()
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure Selenium is installed: pip install selenium")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_browser_html()
