"""JNNX package capabilities and synthetic-likelihood configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

KNOWN_CAPABILITIES = frozenset({"emulator", "synthetic_likelihood"})
DEFAULT_CAPABILITIES = ["emulator"]


def get_capabilities(metadata: Dict[str, Any]) -> List[str]:
    """Return normalized capability list; absent field defaults to emulator-only."""
    raw = metadata.get("capabilities")
    if raw is None:
        return list(DEFAULT_CAPABILITIES)
    if not isinstance(raw, list) or not raw:
        raise ValueError("capabilities must be a non-empty list when present")
    normalized: List[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("capabilities entries must be non-empty strings")
        name = item.strip()
        if name not in KNOWN_CAPABILITIES:
            raise ValueError(f"unknown capability: {name}")
        if name not in normalized:
            normalized.append(name)
    if "emulator" not in normalized:
        raise ValueError("capabilities must include emulator")
    return normalized


def has_capability(metadata: Dict[str, Any], capability: str) -> bool:
    return capability in get_capabilities(metadata)


def validate_capabilities(metadata: Dict[str, Any]) -> List[str]:
    """Validate capability declarations and cross-field requirements."""
    errors: List[str] = []
    try:
        caps = get_capabilities(metadata)
    except ValueError as exc:
        return [str(exc)]

    unknown = set(metadata.get("capabilities") or []) - KNOWN_CAPABILITIES
    for name in sorted(unknown):
        errors.append(f"unknown capability: {name}")

    if "synthetic_likelihood" in caps:
        if "synthetic_likelihood" not in metadata:
            errors.append(
                "synthetic_likelihood capability requires synthetic_likelihood config block"
            )
    elif "synthetic_likelihood" in metadata:
        errors.append(
            "synthetic_likelihood config block present but capability not declared"
        )

    return errors


def load_likelihood(package_dir: Path) -> Dict[str, Any]:
    path = package_dir / "likelihood.json"
    if not path.exists():
        raise FileNotFoundError(f"likelihood.json not found in {package_dir}")
    with open(path, "r") as f:
        return json.load(f)


def upper_tri_index_pairs(p: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(p) for j in range(i, p)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sl_config(metadata: Dict[str, Any], package_dir: Path) -> Dict[str, Any]:
    """Build codegen config dict for packages with synthetic_likelihood capability."""
    if not has_capability(metadata, "synthetic_likelihood"):
        raise ValueError("package does not declare synthetic_likelihood capability")
    sl = metadata["synthetic_likelihood"]
    like = load_likelihood(package_dir)
    p = int(sl["n_summaries"])
    if int(like.get("n_summaries", p)) != p:
        raise ValueError("likelihood.json n_summaries mismatch with metadata")
    n_chol = p * (p + 1) // 2
    d = len(metadata["input_parameters"])
    sigma = like["sigma_emu"]
    flat_sigma = [float(v) for row in sigma for v in row]
    pairs = upper_tri_index_pairs(p)
    model_name = metadata.get("model_name", metadata["module_name"])
    slug = model_name
    debug = metadata.get("debug_exports") or {}
    return {
        "p": p,
        "n_chol": n_chol,
        "d": d,
        "m_out": p + n_chol,
        "distribution_name": sl["distribution_name"],
        "predict_name": f"{slug}_predict",
        "emulator_name": metadata["function_name"],
        "mean_name": f"{slug}_mean",
        "omega1_name": f"{slug}_omega1",
        "omega_total_name": f"{slug}_omega_total",
        "sigma_emu_flat": flat_sigma,
        "upper_tri_pairs": pairs,
        "debug_exports": {
            "predict": debug.get("predict", True),
            "mean": debug.get("mean", True),
            "omega1": debug.get("omega1", True),
            "omega_total": debug.get("omega_total", True),
        },
        "onnx_sha256": sha256_file(package_dir / "model.onnx"),
        "likelihood_sha256": sha256_file(package_dir / "likelihood.json"),
    }


def validate_sl_package(metadata: Dict[str, Any], package_dir: Path) -> List[str]:
    """Return validation errors for synthetic_likelihood capability packages."""
    errors: List[str] = []
    if not has_capability(metadata, "synthetic_likelihood"):
        return errors

    sl = metadata.get("synthetic_likelihood") or {}
    required_sl = [
        "n_summaries",
        "onnx_layout",
        "distribution_name",
        "trial_count_arg",
        "include_sigma_emu",
    ]
    for field in required_sl:
        if field not in sl:
            errors.append(f"Missing synthetic_likelihood.{field}")

    if sl.get("onnx_layout") not in (None, "concatenated", "split"):
        errors.append("synthetic_likelihood.onnx_layout must be concatenated or split")

    p = sl.get("n_summaries")
    if p is not None:
        n_chol = int(p) * (int(p) + 1) // 2
        expected_m = int(p) + n_chol
        outputs = metadata.get("output_parameters", [])
        if len(outputs) != expected_m:
            errors.append(
                f"output_parameters length must be {expected_m} for p={p}, got {len(outputs)}"
            )

    if sl.get("include_sigma_emu"):
        like_path = package_dir / "likelihood.json"
        if not like_path.exists():
            errors.append("likelihood.json required when include_sigma_emu is true")
        else:
            try:
                like = load_likelihood(package_dir)
                sigma = like.get("sigma_emu")
                if p is not None and sigma is not None:
                    if len(sigma) != int(p) or any(len(row) != int(p) for row in sigma):
                        errors.append("sigma_emu must be p x p in likelihood.json")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                errors.append(f"Invalid likelihood.json: {exc}")

    return errors
