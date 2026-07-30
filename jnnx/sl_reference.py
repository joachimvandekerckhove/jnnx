"""Reference SL math (Python mirror of jnnx/cpp/sl_math and compute_sl_logdens_ref)."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

# Matches jnnx/cpp/sl_math.cpp kJitter
JITTER = 1e-10

TransformName = str

_FORWARD: Dict[TransformName, Callable[[float], float]] = {
    "identity": lambda y: float(y),
    "log1p": lambda y: float(np.log1p(y)),
    "log": lambda y: float(np.log(y)),
    "sqrt": lambda y: float(np.sqrt(y)),
}

_INVERSE: Dict[TransformName, Callable[[float], float]] = {
    "identity": lambda z: float(z),
    "log1p": lambda z: float(np.expm1(z)),
    "log": lambda z: float(np.exp(z)),
    "sqrt": lambda z: float(z * z),
}


def _valid_raw(transform: TransformName, y: float) -> bool:
    if not np.isfinite(y):
        return False
    if transform == "log1p":
        return y >= -1.0
    if transform == "log":
        return y > 0.0
    if transform == "sqrt":
        return y >= 0.0
    return True


def obs_raw_to_std(
    raw: Sequence[float],
    transforms: Sequence[TransformName],
    mean: Sequence[float],
    scale: Sequence[float],
) -> Tuple[np.ndarray, bool]:
    """Forward transform raw observations to standardized space."""
    p = len(raw)
    if len(transforms) != p or len(mean) != p or len(scale) != p:
        return np.zeros(p, dtype=np.float64), False
    out = np.zeros(p, dtype=np.float64)
    for j in range(p):
        s = float(scale[j])
        if s == 0.0 or not np.isfinite(s) or not np.isfinite(mean[j]):
            return out, False
        y = float(raw[j])
        t = transforms[j]
        if t not in _FORWARD or not _valid_raw(t, y):
            return out, False
        z = _FORWARD[t](y)
        if not np.isfinite(z):
            return out, False
        out[j] = (z - float(mean[j])) / s
        if not np.isfinite(out[j]):
            return out, False
    return out, True


def obs_std_to_raw(
    std_in: Sequence[float],
    transforms: Sequence[TransformName],
    mean: Sequence[float],
    scale: Sequence[float],
) -> Tuple[np.ndarray, bool]:
    """Inverse transform standardized observations to raw physical space."""
    p = len(std_in)
    if len(transforms) != p or len(mean) != p or len(scale) != p:
        return np.zeros(p, dtype=np.float64), False
    out = np.zeros(p, dtype=np.float64)
    for j in range(p):
        s = float(scale[j])
        if s == 0.0 or not np.isfinite(s) or not np.isfinite(mean[j]):
            return out, False
        z = float(std_in[j]) * s + float(mean[j])
        if not np.isfinite(z):
            return out, False
        t = transforms[j]
        if t not in _INVERSE:
            return out, False
        y = _INVERSE[t](z)
        if not np.isfinite(y) or not _valid_raw(t, y):
            return out, False
        out[j] = y
    return out, True


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
    quad = float(diff @ omega_work @ diff)
    return float(-0.5 * (p * np.log(2.0 * np.pi) - logdet + quad))
