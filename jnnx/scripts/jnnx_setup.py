#!/usr/bin/env python3
"""
JNNX Setup Tool - Configure and edit .jnnx packages
"""

import sys
import os
import json
from pathlib import Path

def main():
    """Main entry point for jnnx-setup command."""
    if len(sys.argv) != 2:
        print("Usage: jnnx-setup <package.jnnx>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    
    # Add the parent directory to Python path to import from jnnx
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from jnnx.core import JNNXPackage
    
    try:
        package = JNNXPackage(package_path)
        print(f"Package: {package.model_name}")
        print(f"Input dimension: {package.input_dim}")
        print(f"Output dimension: {package.output_dim}")
        print(f"Metadata: {package.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
