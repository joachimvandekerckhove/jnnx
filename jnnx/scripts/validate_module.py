#!/usr/bin/env python3
"""
JNNX Module Validation Tool - Test compiled JAGS modules
"""

import sys
from pathlib import Path

def main():
    """Main entry point for validate-module command."""
    if len(sys.argv) != 2:
        print("Usage: validate-module <package.jnnx>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    
    # Add the parent directory to Python path to import from jnnx
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from jnnx.core import JNNXPackage
    
    try:
        package = JNNXPackage(package_path)
        print(f"Validating module for {package.model_name}")
        
        # Validate the package first
        is_valid, errors = package.validate()
        if not is_valid:
            print("Package validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print("Package validation: PASSED")
        print("Module validation would proceed here...")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
