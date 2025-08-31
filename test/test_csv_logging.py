#!/usr/bin/env python3
"""
Test script to demonstrate CSV logging for concurrent operations.
"""

import logging
import sys
import time
import threading
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yt_search_dl.utils import configure_logging, log_download_result_csv, get_csv_log_file


def simulate_concurrent_downloads():
    """Simulate concurrent download operations with CSV logging."""
    
    # Configure logging with CSV enabled
    log_file = Path("logs/test_csv.log")
    configure_logging("INFO", log_file, enable_csv_logging=True)
    
    csv_file = get_csv_log_file()
    if csv_file:
        print(f"✓ CSV logging enabled: {csv_file}")
    else:
        print("✗ CSV logging not enabled")
        return
    
    # Simulate some concurrent operations
    def worker(worker_id: int, query: str, delay: float, success: bool):
        """Simulate a worker processing a download."""
        query_index = worker_id
        
        # Log start
        log_download_result_csv(query, None, False, None, query_index, None, "started")
        print(f"Worker {worker_id}: Started processing '{query}'")
        
        # Simulate work
        time.sleep(delay)
        
        if success:
            # Simulate success
            url = f"https://youtube.com/watch?v=example{worker_id}"
            duration_ms = delay * 1000
            log_download_result_csv(query, url, True, None, query_index, duration_ms, "completed")
            print(f"Worker {worker_id}: Completed '{query}' -> {url}")
        else:
            # Simulate failure
            reason = f"Simulated failure for worker {worker_id}"
            duration_ms = delay * 1000
            log_download_result_csv(query, None, False, reason, query_index, duration_ms, "failed")
            print(f"Worker {worker_id}: Failed '{query}' - {reason}")
    
    # Create test data
    test_queries = [
        "Shape of You by Ed Sheeran",
        "Blinding Lights by The Weeknd", 
        "Bad Guy by Billie Eilish",
        "Dance Monkey by Tones and I",
        "Someone You Loved by Lewis Capaldi"
    ]
    
    # Start concurrent workers
    threads = []
    for i, query in enumerate(test_queries, 1):
        # Alternate success/failure for demonstration
        success = i % 2 == 0
        delay = 0.5 + (i * 0.2)  # Varying delays
        
        thread = threading.Thread(
            target=worker,
            args=(i, query, delay, success),
            name=f"Worker-{i}"
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    
    print(f"\n✓ All workers completed. Check CSV log: {csv_file}")
    print("\nCSV log contents:")
    
    # Display CSV contents
    try:
        with csv_file.open('r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num == 1:  # Header
                    print(f"  {line.strip()}")
                else:
                    print(f"  {line.strip()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")


if __name__ == "__main__":
    print("Testing CSV logging for concurrent operations...")
    simulate_concurrent_downloads()
