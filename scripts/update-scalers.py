#!/usr/bin/env python3
"""Root wrapper for packaged update-scalers implementation."""

import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from jnnx.scripts.update_scalers import main as entry
    entry()


if __name__ == "__main__":
    main()
