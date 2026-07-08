"""Baked sigma_emu helpers for SL module validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_sigma_emu_from_likelihood(package_dir: Path) -> np.ndarray:
    """Full-precision sigma_emu from likelihood.json."""
    path = package_dir / "likelihood.json"
    if not path.exists():
        raise FileNotFoundError(f"likelihood.json not found in {package_dir}")
    with open(path) as f:
        payload = json.load(f)
    return np.asarray(payload["sigma_emu"], dtype=np.float64)


def load_sigma_emu_baked(build_dir: Path) -> np.ndarray:
    """sigma_emu values baked into the compiled module (from build_manifest.json)."""
    manifest_path = build_dir / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"build_manifest.json not found in {build_dir}; run generate-module first"
        )
    manifest = json.loads(manifest_path.read_text())
    flat = manifest.get("sigma_emu_baked")
    if flat is None:
        raise KeyError("build_manifest.json missing sigma_emu_baked")
    p = int(manifest.get("sl_p", int(len(flat) ** 0.5)))
    if len(flat) != p * p:
        raise ValueError(f"sigma_emu_baked length {len(flat)} != p*p for p={p}")
    return np.asarray(flat, dtype=np.float64).reshape(p, p)


def load_sigma_emu_for_validation(
    package_dir: Path,
    build_dir: Optional[Path] = None,
) -> np.ndarray:
    """Prefer baked sigma from build manifest when validating a compiled module."""
    if build_dir is not None:
        manifest = build_dir / "build_manifest.json"
        if manifest.exists():
            try:
                return load_sigma_emu_baked(build_dir)
            except (KeyError, ValueError, FileNotFoundError):
                pass
    return load_sigma_emu_from_likelihood(package_dir)
