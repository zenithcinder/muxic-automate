#!/usr/bin/env python3
"""
Test script to verify GPU detection for the embed model.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from yt_search_dl.matching import _get_preferred_device

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gpu_detection():
    """Test the GPU detection functionality."""
    print("Testing GPU detection for embed model...")
    
    try:
        device = _get_preferred_device()
        print(f"✓ Preferred device: {device}")
        
        if device == "cuda":
            print("✓ GPU acceleration will be used for AI matching")
        else:
            print("✓ CPU will be used for AI matching")
            
        # Test if we can import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("✓ sentence-transformers is available")
            
            # Test model loading on the detected device
            print(f"Testing model loading on {device}...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            print(f"✓ Successfully loaded model on {device}")
            
            # Test encoding
            test_text = "Hello world"
            embedding = model.encode([test_text])
            print(f"✓ Successfully encoded text on {device} (embedding shape: {embedding.shape})")
            
        except ImportError:
            print("✗ sentence-transformers not installed")
            print("  Install with: pip install sentence-transformers numpy")
        except Exception as e:
            print(f"✗ Error testing model: {e}")
            
    except Exception as e:
        print(f"✗ Error in GPU detection: {e}")

if __name__ == "__main__":
    test_gpu_detection()
