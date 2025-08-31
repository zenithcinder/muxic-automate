#!/usr/bin/env python3
"""
Test script to verify GPU detection for the embed model.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yt_search_dl.matching import _get_preferred_device, check_gpu_availability

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gpu_detection():
    """Test the GPU detection functionality."""
    print("Testing GPU detection for embed model...")
    
    # First, get detailed GPU diagnostics
    print("\n=== GPU Diagnostics ===")
    diagnostics = check_gpu_availability()
    
    if diagnostics["cuda_available"]:
        if diagnostics["gpu_available"]:
            print(f"✓ CUDA GPU detected: {diagnostics['gpu_name']}")
            if diagnostics["gpu_memory"]:
                print(f"  Memory: {diagnostics['gpu_memory']['allocated_gb']}GB allocated, {diagnostics['gpu_memory']['reserved_gb']}GB reserved")
        else:
            print(f"✗ CUDA available but GPU test failed: {diagnostics['error']}")
    else:
        print("✗ No CUDA GPU available")
    
    print(f"\n=== Device Selection ===")
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
            print(f"\n=== Model Loading Test ===")
            print(f"Testing model loading on {device}...")
            # Handle meta tensor issue with newer PyTorch versions
            try:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
            except RuntimeError as meta_error:
                if "meta tensor" in str(meta_error).lower():
                    # Use to_empty() for meta tensor handling
                    print("  Handling meta tensor issue with to_empty() method...")
                    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    model = model.to_empty(device=device)
                else:
                    raise
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
