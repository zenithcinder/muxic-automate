        # Create a test configuration with Google filter enabled
        test_config = RuntimeConfig(
            filter_queries_with_google=True,
            use_simple_web_scraping_fallback=True,
            use_browser_based_search=True,  # Enable browser-based search
            google_filter_min_score=15.0,  # Lower threshold for testing
        )
        
        # Set the runtime configuration
        set_runtime_config(test_config)
        
        print("Configuration set:")
        print(f"  filter_queries_with_google: {test_config.filter_queries_with_google}")
        print(f"  use_simple_web_scraping_fallback: {test_config.use_simple_web_scraping_fallback}")
        print(f"  use_browser_based_search: {test_config.use_browser_based_search}")
        print(f"  google_filter_min_score: {test_config.google_filter_min_score}")
        print()
