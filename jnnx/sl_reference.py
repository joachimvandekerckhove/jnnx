"""Reference SL math (Python mirror of jnnx/cpp/sl_math and compute_sl_logdens_ref)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Matches jnnx/cpp/sl_math.cpp kJitter
JITTER = 1e-10


def upper_tri_index_pairs(p: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(p) for j in range(i, p)]


def _add_jitter(mat: np.ndarray, p: int, apply_jitter: bool) -> np.ndarray:
    if not apply_jitter:
        return mat
    out = mat.copy()
    for i in range(p):
        out[i, i] += JITTER
    return out


def omega1_from_chol_upper(chol_upper: Sequence[float], p: int) -> np.ndarray:
    L = np.zeros((p, p), dtype=np.float64)
    for k, (i, j) in enumerate(upper_tri_index_pairs(p)):
        L[i, j] = chol_upper[k]
    return L.T @ L


def omega_total_from_chol(
    chol_upper: Sequence[float],
    n_trials: float,
    sigma_emu: np.ndarray,
    p: int,
    *,
    apply_jitter: bool = True,
) -> np.ndarray:
    omega1 = omega1_from_chol_upper(chol_upper, p)
    sigma_sampling = np.linalg.inv(n_trials * omega1)
    sigma_total = sigma_sampling + sigma_emu
    sigma_total = _add_jitter(sigma_total, p, apply_jitter)
    return np.linalg.inv(sigma_total)


def mvn_logdens_precision(
    x: Sequence[float],
    mu: Sequence[float],
    omega: np.ndarray,
    p: int,
    *,
    apply_jitter: bool = True,
) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    omega_work = _add_jitter(np.asarray(omega, dtype=np.float64), p, apply_jitter)
    sign, logdet = np.linalg.slogdet(omega_work)
    if sign <= 0:
        return float("-inf")
    quad = float(diff @ np.linalg.solve(omega_work, diff))
    return float(-0.5 * (p * np.log(2.0 * np.pi) - logdet + quad))
