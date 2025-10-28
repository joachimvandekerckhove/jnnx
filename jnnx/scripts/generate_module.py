#!/usr/bin/env python3
"""
JNNX Module Generation Tool - Generate C++ JAGS modules from .jnnx packages
"""

import sys
from pathlib import Path

def main():
    """Main entry point for generate-module command."""
    if len(sys.argv) != 2:
        print("Usage: generate-module <package.jnnx>")
        sys.exit(1)
    
    package_path = sys.argv[1]
    
    # Add the parent directory to Python path to import from jnnx
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from jnnx.core import JNNXPackage, JAGSModule
    
    try:
        package = JNNXPackage(package_path)
        build_dir = f"tmp/{package.model_name}.jnnx_build"
        
        module = JAGSModule(package, build_dir)
        print(f"Generating module for {package.model_name}")
        print(f"Build directory: {build_dir}")
        
        # For now, just validate the package
        is_valid, errors = package.validate()
        if not is_valid:
            print("Package validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print("Package validation: PASSED")
        print("Module generation would proceed here...")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
