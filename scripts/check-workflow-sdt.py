#!/usr/bin/env python3
"""
Integration smoke check for demos/workflow-sdt.ipynb.

This check is intentionally lightweight (no notebook execution) and validates
that the notebook still contains expected integration content for the JNNX
workflow.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    nb_path = repo_root / "demos" / "workflow-sdt.ipynb"

    if not nb_path.exists():
        print(f"Error: notebook not found: {nb_path}")
        sys.exit(1)

    try:
        notebook = json.loads(nb_path.read_text())
    except Exception as exc:
        print(f"Error: failed to parse notebook JSON: {exc}")
        sys.exit(1)

    cells = notebook.get("cells", [])
    if len(cells) < 10:
        print(f"Error: notebook has too few cells ({len(cells)})")
        sys.exit(1)

    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )

    required_snippets = [
        "py2jags.run_jags(",
        "onnxruntime",
        "InferenceSession",
        "jnnx",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in code_text]
    if missing:
        print(f"Error: workflow notebook is missing expected snippets: {missing}")
        sys.exit(1)

    print(f"OK: workflow-sdt notebook smoke check passed ({len(cells)} cells)")


if __name__ == "__main__":
    main()
