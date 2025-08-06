#!/usr/bin/env python3
"""Test script to verify the cache clearing functionality"""

import sys
from pathlib import Path

# Add the current directory to Python path to import from build_docs
sys.path.insert(0, '.')

# Import the function from build_docs
from build_docs import clear_quarto_cache

if __name__ == "__main__":
    print("Testing cache clearing functionality...")
    clear_quarto_cache()
    print("Cache clearing test completed!")