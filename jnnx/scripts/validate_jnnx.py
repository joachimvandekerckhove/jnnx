#!/usr/bin/env python3
"""
JNNX Validation Tool - Validate .jnnx packages
"""

import sys
from pathlib import Path

def main():
    """Main entry point for validate-jnnx command."""
    if len(sys.argv) != 2:
        print("Usage: validate-jnnx <package.jnnx>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    
    # Add the parent directory to Python path to import from jnnx
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from jnnx.core import validate_jnnx_package
    
    try:
        is_valid, errors = validate_jnnx_package(package_path)
        
        if is_valid:
            print("Package validation: PASSED")
        else:
            print("Package validation: FAILED")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
