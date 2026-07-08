#!/usr/bin/env python3
"""
Reference synthetic-likelihood log-density for JNNX v1.1 validation.

Implements the same algebra as build_mv_jags_likelihood_lines in esl/mv.py:

    Omega1 = L^T L
    Sigma_sampling = inv(N * Omega1)
    Sigma_total = Sigma_sampling + sigma_emu
    log p(obs_std | theta, N) = log MVN(obs_std; mu_std, Omega_total)

where dmnorm / JAGS use precision Omega_total = inv(Sigma_total).

Usage:
    # Single evaluation
    python scripts/compute_sl_logdens_ref.py \\
        --package models/ddm3mv.jnnx \\
        --theta -0.5 1.2 0.25 --n-trials 600 \\
        --obs-std 0.1 -0.2 0.05

    # Write regression fixture for validate-sl (requires ESL training repo)
    python scripts/compute_sl_logdens_ref.py \\
        --package models/ddm3mv.jnnx \\
        --write-fixture fixtures/ddm3mv_sl_regression.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

PARAM_BOUNDS = ((-2.0, 2.0), (0.5, 2.0), (0.15, 0.45))
PARAM_NAMES = ("v", "a", "t0")
N_SUMMARIES = 3
DEFAULT_N_TRIALS = (200, 400, 600, 800)


def upper_tri_index_pairs(p: int) -> list[tuple[int, int]]:
    """Row-major upper triangle index pairs."""
    return [(i, j) for i in range(p) for j in range(i, p)]


def load_sigma_emu(package_dir: Path) -> np.ndarray:
    """Load sigma_emu from likelihood.json in a .jnnx package."""
    path = package_dir / "likelihood.json"
    if not path.exists():
        raise FileNotFoundError(f"likelihood.json not found in {package_dir}")
    with open(path) as f:
        payload = json.load(f)
    return np.asarray(payload["sigma_emu"], dtype=np.float64)


def load_n_summaries(package_dir: Path) -> int:
    """Load n_summaries from likelihood.json or metadata."""
    like_path = package_dir / "likelihood.json"
    if like_path.exists():
        with open(like_path) as f:
            payload = json.load(f)
        if "n_summaries" in payload:
            return int(payload["n_summaries"])
    meta_path = package_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        sl = meta.get("synthetic_likelihood", {})
        if sl.get("n_summaries"):
            return int(sl["n_summaries"])
    return N_SUMMARIES


def omega1_from_chol_upper(chol_upper: np.ndarray, n: int) -> np.ndarray:
    """Per-trial precision Omega1 = L^T L from flat upper-triangular chol."""
    L = np.zeros((n, n), dtype=np.float64)
    for k, (i, j) in enumerate(upper_tri_index_pairs(n)):
        L[i, j] = chol_upper[k]
    return L.T @ L


def omega_total_from_chol(
    chol_upper: np.ndarray,
    n_trials: float,
    sigma_emu: np.ndarray,
    n: int,
) -> np.ndarray:
    """Final precision matrix passed to dmnorm / {name}_sl logDensity."""
    omega1 = omega1_from_chol_upper(chol_upper, n)
    sigma_sampling = np.linalg.inv(n_trials * omega1)
    sigma_total = sigma_sampling + sigma_emu
    return np.linalg.inv(sigma_total)


def mvn_logdens_precision(
    x: np.ndarray,
    mu: np.ndarray,
    omega: np.ndarray,
    n: int,
) -> float:
    """Multivariate normal log-density with precision matrix (JAGS dmnorm form)."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0:
        raise ValueError("Omega_total is not positive definite")
    quad = float(diff @ np.linalg.solve(omega, diff))
    return float(-0.5 * (n * np.log(2.0 * np.pi) - logdet + quad))


def onnx_predict(
    session: ort.InferenceSession, theta: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX once; return (mu_std, chol_upper)."""
    x = np.asarray(theta, dtype=np.float32).reshape(1, -1)
    out = session.run(None, {"input": x})[0][0]
    return out[:n].astype(np.float64), out[n:].astype(np.float64)


def sl_logdens(
    obs_std: np.ndarray,
    theta: np.ndarray,
    n_trials: float,
    session: ort.InferenceSession,
    sigma_emu: np.ndarray,
    n: int = N_SUMMARIES,
) -> float:
    """Integrated synthetic-likelihood log-density at standardized summaries."""
    mu_std, chol_upper = onnx_predict(session, theta, n)
    omega_total = omega_total_from_chol(chol_upper, n_trials, sigma_emu, n)
    return mvn_logdens_precision(obs_std, mu_std, omega_total, n)


def sl_logdens_decomposed(
    obs_std: np.ndarray,
    theta: np.ndarray,
    n_trials: float,
    session: ort.InferenceSession,
    sigma_emu: np.ndarray,
    n: int = N_SUMMARIES,
) -> dict:
    """Return log-density plus intermediates for QA."""
    mu_std, chol_upper = onnx_predict(session, theta, n)
    omega1 = omega1_from_chol_upper(chol_upper, n)
    sigma_sampling = np.linalg.inv(n_trials * omega1)
    sigma_total = sigma_sampling + sigma_emu
    omega_total = np.linalg.inv(sigma_total)
    logdens = mvn_logdens_precision(obs_std, mu_std, omega_total, n)
    return {
        "logdens": logdens,
        "mu_std": mu_std,
        "chol_upper": chol_upper,
        "omega1": omega1,
        "sigma_sampling": sigma_sampling,
        "sigma_total": sigma_total,
        "omega_total": omega_total,
    }


def build_regression_cases(
    package_dir: Path,
    *,
    n_cases: int = 100,
    seed: int = 42,
    n_trials_choices: tuple[int, ...] = DEFAULT_N_TRIALS,
) -> list[dict]:
    """Build (theta, n_trials, obs_std, logdens) regression rows (requires ESL)."""
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from esl.data import load_target_transform  # noqa: WPS433
        from models.ddm.ddm3mv import DDM3MV  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit(
            "Fixture regeneration requires the ESL training repository "
            "(esl, models.ddm.ddm3mv). Use the committed "
            "fixtures/ddm3mv_sl_regression.json in JNNX-only checkouts."
        ) from exc

    n = load_n_summaries(package_dir)
    onnx_path = package_dir / "model.onnx"
    sigma_emu = load_sigma_emu(package_dir)
    session = ort.InferenceSession(str(onnx_path))

    model = DDM3MV
    transform = load_target_transform("ddm3mv")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    attempts = 0
    while len(rows) < n_cases and attempts < n_cases * 5:
        attempts += 1
        theta = np.array([rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS])
        n_trials = int(rng.choice(n_trials_choices))
        summaries = model.simulate_summaries(theta, n_trials, 10_000 + len(rows))
        if not np.all(np.isfinite(summaries)):
            continue
        obs_std = transform.transform(summaries.reshape(1, -1))[0]
        logdens = sl_logdens(obs_std, theta, n_trials, session, sigma_emu, n=n)
        rows.append(
            {
                "theta": theta.tolist(),
                "n_trials": n_trials,
                "obs_std": obs_std.tolist(),
                "logdens": logdens,
            }
        )

    if len(rows) < n_cases:
        raise RuntimeError(f"Only generated {len(rows)} of {n_cases} regression cases")
    return rows


def write_fixture(path: Path, package_dir: Path, rows: list[dict]) -> None:
    """Write validate-sl regression fixture JSON."""
    n = load_n_summaries(package_dir)
    onnx_bytes = (package_dir / "model.onnx").read_bytes()

    payload = {
        "version": "1.0",
        "slug": "ddm3mv",
        "package_path": str(package_dir.relative_to(ROOT))
        if package_dir.is_relative_to(ROOT)
        else str(package_dir),
        "distribution_name": "ddm3mv_sl",
        "variant": "n_agnostic_cholesky",
        "n_summaries": n,
        "param_names": list(PARAM_NAMES),
        "param_bounds": [list(b) for b in PARAM_BOUNDS],
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "likelihood_sha256": hashlib.sha256(
            (package_dir / "likelihood.json").read_bytes()
        ).hexdigest(),
        "tolerance": {"atol": 1e-4, "rtol": 1e-5},
        "cases": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[compute_sl_logdens_ref] Wrote {len(rows)} cases to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=ROOT / "models" / "ddm3mv.jnnx",
        help="Path to .jnnx package directory",
    )
    parser.add_argument(
        "--theta",
        type=float,
        nargs=3,
        metavar=("V", "A", "T0"),
        help="Model parameters (v, a, t0)",
    )
    parser.add_argument("--n-trials", type=float, help="Trial count N")
    parser.add_argument(
        "--obs-std",
        type=float,
        nargs=3,
        metavar=("S1", "S2", "S3"),
        help="Observed standardized summary vector",
    )
    parser.add_argument(
        "--write-fixture",
        type=Path,
        metavar="PATH",
        help="Write regression fixture JSON to PATH",
    )
    parser.add_argument("--n-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decomposed", action="store_true")
    return parser.parse_args()


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    package_dir = args.package.resolve()
    if not (package_dir / "model.onnx").exists():
        raise FileNotFoundError(f"model.onnx not found in {package_dir}")

    if args.write_fixture:
        rows = build_regression_cases(
            package_dir, n_cases=args.n_cases, seed=args.seed
        )
        write_fixture(args.write_fixture.resolve(), package_dir, rows)
        return

    if args.theta is None or args.n_trials is None or args.obs_std is None:
        raise SystemExit(
            "Provide --theta, --n-trials, and --obs-std, or use --write-fixture"
        )

    n = load_n_summaries(package_dir)
    sigma_emu = load_sigma_emu(package_dir)
    session = ort.InferenceSession(str(package_dir / "model.onnx"))
    theta = np.array(args.theta, dtype=np.float64)
    obs_std = np.array(args.obs_std, dtype=np.float64)

    if args.decomposed:
        result = sl_logdens_decomposed(
            obs_std, theta, args.n_trials, session, sigma_emu, n=n
        )
        print(json.dumps(_jsonify(result), indent=2))
    else:
        logdens = sl_logdens(obs_std, theta, args.n_trials, session, sigma_emu, n=n)
        print(f"logdens = {logdens:.12f}")


if __name__ == "__main__":
    main()
