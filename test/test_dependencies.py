#!/usr/bin/env python3
"""
Test script to verify all dependencies can be imported correctly.
"""

import sys
from pathlib import Path


def test_core_dependencies():
    """Test core dependencies that are always required."""
    print("Testing core dependencies...")

    try:
        import yt_dlp

        print(f"✓ yt-dlp {yt_dlp.version.__version__}")
    except ImportError as e:
        print(f"✗ yt-dlp: {e}")
        return False

    return True


def test_ai_dependencies():
    """Test AI matching dependencies (optional)."""
    print("\nTesting AI matching dependencies...")

    dependencies = [
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("huggingface-hub", "huggingface_hub"),
        ("safetensors", "safetensors"),
    ]

    all_available = True
    for package_name, import_name in dependencies:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package_name} {version}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            all_available = False

    return all_available


def test_utility_dependencies():
    """Test utility dependencies."""
    print("\nTesting utility dependencies...")

    dependencies = [
        ("requests", "requests"),
        ("urllib3", "urllib3"),
        ("certifi", "certifi"),
        ("charset-normalizer", "charset_normalizer"),
        ("idna", "idna"),
        ("fsspec", "fsspec"),
        ("filelock", "filelock"),
    ]

    all_available = True
    for package_name, import_name in dependencies:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package_name} {version}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            all_available = False

    return all_available


def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")

    try:
        from yt_search_dl import RuntimeConfig, process_queries

        print("✓ yt_search_dl package")
    except ImportError as e:
        print(f"✗ yt_search_dl package: {e}")
        return False

    return True


def main():
    """Run all dependency tests."""
    print("Dependency Test Results")
    print("=" * 50)

    core_ok = test_core_dependencies()
    ai_ok = test_ai_dependencies()
    util_ok = test_utility_dependencies()
    project_ok = test_project_modules()

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Core dependencies: {'✓' if core_ok else '✗'}")
    print(f"AI dependencies: {'✓' if ai_ok else '✗'}")
    print(f"Utility dependencies: {'✓' if util_ok else '✗'}")
    print(f"Project modules: {'✓' if project_ok else '✗'}")

    if core_ok and project_ok:
        print("\n✅ Core functionality should work!")
        if ai_ok:
            print("✅ AI matching is available!")
        else:
            print(
                "⚠️  AI matching is not available (install with: pip install -r requirements.txt)"
            )
    else:
        print("\n❌ Core functionality may not work properly")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
