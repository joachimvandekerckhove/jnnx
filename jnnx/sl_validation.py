"""Synthetic-likelihood validation suite for validate-module."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from jnnx.capabilities import build_sl_config, has_capability, sha256_file
from jnnx.core import JNNXPackage
from jnnx.sl_reference import (
    mvn_logdens_precision,
    obs_raw_to_std,
    omega1_from_chol_upper,
    omega_total_from_chol,
)
from jnnx.sl_sigma import load_sigma_emu_for_validation

ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return ROOT


def resolve_fixture_path(
    package_dir: Path,
    *,
    fixture_arg: Optional[Path] = None,
) -> Path:
    """Locate SL regression fixture for a package."""
    if fixture_arg is not None:
        path = fixture_arg.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        return path

    env_dir = os.environ.get("JNNX_FIXTURES_DIR")
    slug = package_dir.name.replace(".jnnx", "")
    project = _project_root()
    candidates = [
        project / "fixtures" / f"{slug}_sl_regression.json",
        package_dir.parent / "fixtures" / f"{slug}_sl_regression.json",
        package_dir / f"{slug}_sl_regression.json",
        project / "fixtures" / "ddm3mv_sl_regression.json",
        project / "docs" / "internal" / "fixtures" / "ddm3mv_sl_regression.json",
    ]
    if env_dir:
        candidates.insert(0, Path(env_dir).expanduser() / f"{slug}_sl_regression.json")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Regression fixture not found for {slug}; tried {', '.join(str(p) for p in candidates)}"
    )


def _load_fixture(
    package_dir: Path,
    *,
    fixture_arg: Optional[Path] = None,
) -> Dict[str, Any]:
    path = resolve_fixture_path(package_dir, fixture_arg=fixture_arg)
    return json.loads(path.read_text())


def _run_jags_deterministic(
    module_name: str,
    model_code: str,
    monitor: List[str],
    data: Optional[Dict[str, Any]] = None,
    *,
    extra_modules: Optional[List[str]] = None,
) -> Any:
    import py2jags

    data_dict = dict(data or {})
    if "n" not in data_dict:
        data_dict["n"] = 1
    modules = [module_name]
    if extra_modules:
        for name in extra_modules:
            if name not in modules:
                modules.append(name)
    return py2jags.run_jags(
        model_string=model_code,
        data_dict=data_dict,
        nchains=1,
        nsamples=1,
        nadapt=0,
        nburnin=0,
        monitorparams=monitor,
        modules=modules,
    )


def _obs_transform_args(sl_cfg: Dict[str, Any]) -> Tuple[List[str], List[float], List[float]]:
    baked = sl_cfg["obs_transform_baked"]
    return (
        list(baked["column_transforms"]),
        list(baked["scaler_mean"]),
        list(baked["scaler_scale"]),
    )


def _raw_to_std(
    obs_raw: Sequence[float],
    sl_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, bool]:
    transforms, mean, scale = _obs_transform_args(sl_cfg)
    return obs_raw_to_std(obs_raw, transforms, mean, scale)


def _scipy_mvn_logpdf(
    obs_raw: Sequence[float],
    mu: np.ndarray,
    omega: np.ndarray,
    sl_cfg: Dict[str, Any],
) -> float:
    from scipy.stats import multivariate_normal

    from jnnx.sl_reference import JITTER

    obs_std, ok = _raw_to_std(obs_raw, sl_cfg)
    if not ok:
        raise ValueError("invalid raw observation for transform")

    omega_work = np.asarray(omega, dtype=np.float64).copy()
    p = omega_work.shape[0]
    for i in range(p):
        omega_work[i, i] += JITTER
    cov = np.linalg.inv(omega_work)
    return float(
        multivariate_normal.logpdf(
            obs_std,
            mean=np.asarray(mu, dtype=np.float64),
            cov=cov,
            allow_singular=False,
        )
    )


def _fetch_mu_omega_jags(
    module_name: str,
    mean_name: str,
    omega_total_name: str,
    theta_str: str,
    n_trials: int,
    p: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model_code = f"""
    model {{
        mu[1:{p}] <- {mean_name}({theta_str})
        OmegaTot[1:{p},1:{p}] <- {omega_total_name}({theta_str}, {n_trials})
        dummy ~ dnorm(0, 1) T(0, 0)
    }}
    """
    chains = _run_jags_deterministic(module_name, model_code, ["mu", "OmegaTot"])
    mu = np.array([chains.get_samples(f"mu_{i+1}")[0] for i in range(p)])
    omega = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            omega[i, j] = float(chains.get_samples(f"OmegaTot_{i+1}_{j+1}")[0])
    return mu, omega


def _sl_logdens_ref(
    obs_raw: np.ndarray,
    theta: np.ndarray,
    n_trials: float,
    session: Any,
    sigma_emu: np.ndarray,
    sl_cfg: Dict[str, Any],
    p: int,
) -> float:
    obs_std, ok = _raw_to_std(obs_raw, sl_cfg)
    if not ok:
        return float("-inf")
    out = session.run(None, {"input": np.array([theta], dtype=np.float32)})[0][0]
    mu = out[:p]
    chol = out[p:]
    omega_total = omega_total_from_chol(chol, n_trials, sigma_emu, p)
    return mvn_logdens_precision(obs_std, mu, omega_total, p)


def test_checksums(package: JNNXPackage, fixture: Dict[str, Any]) -> Tuple[bool, str]:
    onnx_h = sha256_file(package.package_path / "model.onnx")
    like_h = package.get_likelihood_sha256()
    obs_h = sha256_file(package.package_path / "obs_transform.json")
    if onnx_h != fixture.get("onnx_sha256"):
        return False, "onnx_sha256 mismatch (stale fixture or package)"
    if like_h != fixture.get("likelihood_sha256"):
        return False, "likelihood_sha256 mismatch (stale sidecar or package)"
    if obs_h != fixture.get("obs_transform_sha256"):
        return False, "obs_transform_sha256 mismatch (stale sidecar or package)"
    return True, "checksums OK"


def test_onnx_layout(package: JNNXPackage, sl_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    session = ort.InferenceSession(package.get_onnx_path())
    out_shape = session.get_outputs()[0].shape
    expected_m = sl_cfg["m_out"]
    if out_shape[-1] != expected_m:
        return False, f"ONNX output dim {out_shape[-1]} != {expected_m}"
    return True, f"ONNX layout M={expected_m}"


def test_predict_parity(package: JNNXPackage, sl_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    metadata = package.metadata
    module_name = metadata["module_name"]
    predict_name = sl_cfg["predict_name"]
    input_params = metadata["input_parameters"]
    theta = [(p["min"] + p["max"]) / 2 for p in input_params]
    theta_str = ", ".join(str(t) for t in theta)

    session = ort.InferenceSession(package.get_onnx_path())
    onnx_out = session.run(None, {"input": np.array([theta], dtype=np.float32)})[0][0]

    model_code = f"""
    model {{
        result[1:{sl_cfg['m_out']}] <- {predict_name}({theta_str})
        dummy ~ dnorm(0, 1)
    }}
    """
    chains = _run_jags_deterministic(module_name, model_code, ["result"])
    jags_out = []
    for i in range(sl_cfg["m_out"]):
        name = f"result_{i+1}"
        if name in chains.parameter_names:
            jags_out.append(chains.get_samples(name)[0])

    if len(jags_out) != sl_cfg["m_out"]:
        return False, f"predict output len {len(jags_out)} != {sl_cfg['m_out']}"

    max_diff = max(abs(a - b) for a, b in zip(onnx_out, jags_out))
    if max_diff > 1e-4:
        return False, f"predict parity max diff {max_diff:.2e}"
    return True, f"predict parity max diff {max_diff:.2e}"


def test_omega1_parity(package: JNNXPackage, sl_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    metadata = package.metadata
    module_name = metadata["module_name"]
    omega1_name = sl_cfg["omega1_name"]
    input_params = metadata["input_parameters"]
    theta = [(p["min"] + p["max"]) / 2 for p in input_params]
    theta_str = ", ".join(str(t) for t in theta)
    p = sl_cfg["p"]

    session = ort.InferenceSession(package.get_onnx_path())
    out = session.run(None, {"input": np.array([theta], dtype=np.float32)})[0][0]
    ref = omega1_from_chol_upper(out[p:], p)

    model_code = f"""
    model {{
        Omega1[1:{p},1:{p}] <- {omega1_name}({theta_str})
        dummy ~ dnorm(0, 1)
    }}
    """
    chains = _run_jags_deterministic(module_name, model_code, ["Omega1"])
    jags_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            name = f"Omega1_{i+1}_{j+1}"
            if name in chains.parameter_names:
                jags_mat[i, j] = chains.get_samples(name)[0]

    max_diff = float(np.max(np.abs(ref - jags_mat)))
    if max_diff > 1e-4:
        return False, f"omega1 parity max diff {max_diff:.2e}"
    return True, f"omega1 parity max diff {max_diff:.2e}"


def test_omega_total_parity(
    package: JNNXPackage,
    sl_cfg: Dict[str, Any],
    build_dir: Optional[Path],
) -> Tuple[bool, str]:
    metadata = package.metadata
    module_name = metadata["module_name"]
    omega_total_name = sl_cfg["omega_total_name"]
    input_params = metadata["input_parameters"]
    theta = [(p["min"] + p["max"]) / 2 for p in input_params]
    theta_str = ", ".join(str(t) for t in theta)
    n_trials = 600
    p = sl_cfg["p"]

    sigma_emu = load_sigma_emu_for_validation(package.package_path, build_dir)

    session = ort.InferenceSession(package.get_onnx_path())
    out = session.run(None, {"input": np.array([theta], dtype=np.float32)})[0][0]
    ref = omega_total_from_chol(out[p:], n_trials, sigma_emu, p)

    model_code = f"""
    model {{
        OmegaTot[1:{p},1:{p}] <- {omega_total_name}({theta_str}, {n_trials})
        dummy ~ dnorm(0, 1)
    }}
    """
    chains = _run_jags_deterministic(module_name, model_code, ["OmegaTot"])
    jags_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            name = f"OmegaTot_{i+1}_{j+1}"
            if name in chains.parameter_names:
                jags_mat[i, j] = chains.get_samples(name)[0]

    max_diff = float(np.max(np.abs(ref - jags_mat)))
    rtol = 1e-6
    rel = max_diff / max(float(np.max(np.abs(ref))), 1.0)
    if max_diff > 1e-3 and rel > rtol:
        return False, f"omega_total parity max diff {max_diff:.2e} rel {rel:.2e}"
    return True, f"omega_total parity max diff {max_diff:.2e} rel {rel:.2e}"


def test_logdens_fixture_sanity(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    build_dir: Optional[Path],
) -> Tuple[bool, str]:
    """8.6a: Python reference with baked sigma_emu vs fixture logdens."""
    from scripts.compute_sl_logdens_ref import sl_logdens

    sigma_emu = load_sigma_emu_for_validation(package.package_path, build_dir)
    session = ort.InferenceSession(package.get_onnx_path())
    atol = fixture["tolerance"]["atol"]
    max_diff = 0.0
    for case in fixture["cases"]:
        ref = case["logdens"]
        got = sl_logdens(
            np.array(case["obs"]),
            np.array(case["theta"]),
            case["n_trials"],
            session,
            sigma_emu,
            package.package_path,
            n=sl_cfg["p"],
        )
        diff = abs(got - ref)
        max_diff = max(max_diff, diff)
        if diff > atol:
            return False, f"fixture sanity diff={diff:.2e} ref={ref} got={got}"
    return True, f"fixture sanity max diff {max_diff:.2e} (n={len(fixture['cases'])})"


def _logdens_jags_expr(
    sl_cfg: Dict[str, Any],
    obs_expr: str,
    theta_str: str,
    n_trials: int,
) -> str:
    logdens_name = sl_cfg["logdens_name"]
    p = sl_cfg["p"]
    if obs_expr == "obs":
        obs_args = ", ".join(f"obs[{i + 1}]" for i in range(p))
    else:
        obs_args = obs_expr
    return f"{logdens_name}({obs_args}, {theta_str}, {n_trials})"


def test_logdens_jags_parity(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    build_dir: Optional[Path],
    *,
    max_cases: int = 10,
) -> Tuple[bool, str]:
    """8.6b: JAGS logdens debug node vs jitter-matched Python reference."""
    metadata = package.metadata
    module_name = metadata["module_name"]
    p = sl_cfg["p"]
    sigma_emu = load_sigma_emu_for_validation(package.package_path, build_dir)
    session = ort.InferenceSession(package.get_onnx_path())
    atol = fixture["tolerance"]["atol"]
    cases = fixture["cases"][:max_cases]
    max_diff = 0.0

    for case in cases:
        theta = case["theta"]
        theta_str = ", ".join(str(t) for t in theta)
        n_trials = case["n_trials"]
        obs = case["obs"]
        ref = _sl_logdens_ref(
            np.array(obs),
            np.array(theta),
            n_trials,
            session,
            sigma_emu,
            sl_cfg,
            p,
        )

        logdens_expr = _logdens_jags_expr(sl_cfg, "obs", theta_str, n_trials)
        model_code = f"""
        model {{
            logdens_hat <- {logdens_expr}
            dummy ~ dnorm(0, 1) T(0, 0)
        }}
        """
        chains = _run_jags_deterministic(
            module_name, model_code, ["logdens_hat"], data={"obs": obs}
        )
        got = float(chains.get_samples("logdens_hat")[0])
        diff = abs(got - ref)
        max_diff = max(max_diff, diff)
        if diff > atol:
            return False, f"JAGS logdens diff={diff:.2e} ref={ref:.6f} got={got:.6f}"

    return True, f"JAGS logdens parity max diff {max_diff:.2e} (n={len(cases)})"


def test_logdens_legacy_subset(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    build_dir: Optional[Path],
    *,
    max_cases: int = 5,
) -> Tuple[bool, str]:
    """8.6c: {name}_logdens vs SciPy dmnorm reference on JAGS mean/omega_total."""
    metadata = package.metadata
    module_name = metadata["module_name"]
    mean_name = sl_cfg["mean_name"]
    omega_total_name = sl_cfg["omega_total_name"]
    p = sl_cfg["p"]
    atol = fixture["tolerance"]["atol"]
    rtol = fixture["tolerance"].get("rtol", 1e-5)
    cases = fixture["cases"][:max_cases]
    max_diff = 0.0

    for case in cases:
        theta = case["theta"]
        theta_str = ", ".join(str(t) for t in theta)
        n_trials = case["n_trials"]
        obs = case["obs"]
        mu, omega = _fetch_mu_omega_jags(
            module_name, mean_name, omega_total_name, theta_str, n_trials, p
        )
        ref = _scipy_mvn_logpdf(obs, mu, omega, sl_cfg)

        logdens_expr = _logdens_jags_expr(sl_cfg, "obs", theta_str, n_trials)
        model_code = f"""
        model {{
            logdens_hat <- {logdens_expr}
            dummy ~ dnorm(0, 1) T(0, 0)
        }}
        """
        chains = _run_jags_deterministic(
            module_name, model_code, ["logdens_hat"], data={"obs": obs}
        )
        got = float(chains.get_samples("logdens_hat")[0])
        diff = abs(got - ref)
        case_atol = max(atol, rtol * max(1.0, abs(ref)), 5e-4)
        max_diff = max(max_diff, diff)
        if diff > case_atol:
            return False, f"SciPy dmnorm diff={diff:.2e} ref={ref:.6f} got={got:.6f}"

    return True, f"SciPy dmnorm parity max diff {max_diff:.2e} (n={len(cases)})"


def test_logdens_dmnorm_parity(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    *,
    max_cases: int = 10,
) -> Tuple[bool, str]:
    """8.10: {name}_logdens vs SciPy MVN (cov=inv(Omega)) on JAGS nodes."""
    metadata = package.metadata
    module_name = metadata["module_name"]
    mean_name = sl_cfg["mean_name"]
    omega_total_name = sl_cfg["omega_total_name"]
    p = sl_cfg["p"]
    atol = fixture["tolerance"]["atol"]
    rtol = fixture["tolerance"].get("rtol", 1e-5)
    cases = fixture["cases"][:max_cases]
    max_diff = 0.0

    for case in cases:
        theta_str = ", ".join(str(t) for t in case["theta"])
        n_trials = case["n_trials"]
        obs = case["obs"]
        mu, omega = _fetch_mu_omega_jags(
            module_name, mean_name, omega_total_name, theta_str, n_trials, p
        )
        ref = _scipy_mvn_logpdf(obs, mu, omega, sl_cfg)

        logdens_expr = _logdens_jags_expr(sl_cfg, "obs", theta_str, n_trials)
        model_code = f"""
        model {{
            logdens_hat <- {logdens_expr}
            dummy ~ dnorm(0, 1) T(0, 0)
        }}
        """
        chains = _run_jags_deterministic(
            module_name, model_code, ["logdens_hat"], data={"obs": obs}
        )
        got = float(chains.get_samples("logdens_hat")[0])
        diff = abs(got - ref)
        case_atol = max(atol, rtol * max(1.0, abs(ref)), 5e-4)
        max_diff = max(max_diff, diff)
        if diff > case_atol:
            return False, f"dmnorm cross-check diff={diff:.2e} ref={ref:.6f} got={got:.6f}"

    return True, f"dmnorm cross-check max diff {max_diff:.2e} (n={len(cases)})"


def test_deviance_sl_vs_legacy(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    *,
    max_cases: int = 3,
) -> Tuple[bool, str]:
    """8.7: {name}_logdens vs JAGS dmnorm deviance (-deviance/2) at fixed theta."""
    metadata = package.metadata
    module_name = metadata["module_name"]
    cases = fixture["cases"][:max_cases]
    max_diff = 0.0
    atol = fixture["tolerance"]["atol"]
    p = sl_cfg["p"]
    mean_name = sl_cfg["mean_name"]
    omega_total_name = sl_cfg["omega_total_name"]

    for case in cases:
        theta_str = ", ".join(str(t) for t in case["theta"])
        n_trials = case["n_trials"]
        obs = case["obs"]
        obs_std, ok = _raw_to_std(obs, sl_cfg)
        if not ok:
            return False, "invalid raw obs in deviance test case"
        logdens_expr = _logdens_jags_expr(sl_cfg, "obs", theta_str, n_trials)

        sl_model = f"""
        model {{
            logdens_sl <- {logdens_expr}
            dummy ~ dnorm(0, 1) T(0, 0)
        }}
        """
        legacy_model = f"""
        model {{
            mu[1:{p}] <- {mean_name}({theta_str})
            OmegaTot[1:{p},1:{p}] <- {omega_total_name}({theta_str}, {n_trials})
            obs_std[1:{p}] ~ dmnorm(mu[1:{p}], OmegaTot[1:{p},1:{p}])
            dummy ~ dnorm(0, 1) T(0, 0)
        }}
        """
        sl_chains = _run_jags_deterministic(
            module_name, sl_model, ["logdens_sl"], data={"obs": obs}
        )
        leg_chains = _run_jags_deterministic(
            module_name,
            legacy_model,
            ["deviance"],
            data={"obs_std": obs_std.tolist()},
            extra_modules=["dic"],
        )
        sl_val = float(sl_chains.get_samples("logdens_sl")[0])
        leg_dev = float(leg_chains.get_samples("deviance")[0])
        leg_val = -0.5 * leg_dev
        diff = abs(sl_val - leg_val)
        max_diff = max(max_diff, diff)
        if diff > atol:
            return False, (
                f"deviance mismatch diff={diff:.2e} logdens={sl_val:.6f} "
                f"legacy={leg_val:.6f}"
            )

    return True, f"logdens vs dmnorm deviance max diff {max_diff:.2e} (n={len(cases)})"


def test_random_sample_ppc(
    package: JNNXPackage,
    sl_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """8.9: randomSample produces finite PPC replicates."""
    import py2jags

    metadata = package.metadata
    module_name = metadata["module_name"]
    dist_name = sl_cfg["distribution_name"]
    p = sl_cfg["p"]
    input_params = metadata["input_parameters"]
    theta = [(par["min"] + par["max"]) / 2 for par in input_params]
    theta_str = ", ".join(str(t) for t in theta)
    n_trials = 600

    model_code = f"""
    model {{
        obs_rep[1:{p}] ~ {dist_name}({theta_str}, {n_trials})
        dummy ~ dnorm(0, 1)
    }}
    """
    chains = py2jags.run_jags(
        model_string=model_code,
        data_dict={"n": 1},
        nchains=1,
        nsamples=20,
        nadapt=0,
        nburnin=0,
        monitorparams=["obs_rep"],
        modules=[module_name],
    )
    samples = []
    for i in range(p):
        name = f"obs_rep_{i+1}"
        if name not in chains.parameter_names:
            return False, f"missing monitor {name}"
        vals = chains.get_samples(name)
        samples.extend(vals)

    if not samples or not np.all(np.isfinite(samples)):
        return False, "randomSample produced non-finite values"

    transforms, _, _ = _obs_transform_args(sl_cfg)
    for i in range(p):
        col_samples = chains.get_samples(f"obs_rep_{i+1}")
        tname = transforms[i]
        if tname == "identity":
            if np.any(col_samples < 0.0) or np.any(col_samples > 1.0):
                return False, f"PPC acc column out of [0,1] range"
        elif tname in ("log1p",):
            if np.any(col_samples < -1.0):
                return False, f"PPC {tname} column below domain (y >= -1)"
        elif tname == "log":
            if np.any(col_samples <= 0.0):
                return False, f"PPC {tname} column has non-positive values"
        elif tname == "sqrt":
            if np.any(col_samples < 0.0):
                return False, "PPC sqrt column has negative values"

    return True, f"PPC randomSample OK ({len(samples)} finite raw draws)"


def test_logdens_raw_obs_parity(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
    build_dir: Optional[Path],
    *,
    max_cases: int = 10,
) -> Tuple[bool, str]:
    """8.11: raw obs through Python transform matches JAGS logdens on raw inputs."""
    return test_logdens_jags_parity(
        package, fixture, sl_cfg, build_dir, max_cases=max_cases
    )


def run_sl_validation(
    package: JNNXPackage,
    build_dir: Optional[Path] = None,
    *,
    fixture_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Run SL capability tests; returns (passed, total)."""
    if not has_capability(package.metadata, "synthetic_likelihood"):
        return 0, 0

    if ort is None:
        print("  [FAIL] SL validation: onnxruntime not available")
        return 0, 1

    sl_cfg = build_sl_config(package.metadata, package.package_path)
    build_dir = build_dir or (_project_root() / "tmp" / f"{package.package_path.name}_build")
    so_path = build_dir / f"{package.metadata['module_name']}.so"
    if not so_path.exists():
        print(f"  Warning: compiled module not found at {so_path}")
        print("  Run generate-module and make before validate-module SL tests.")

    fixture = _load_fixture(package.package_path, fixture_arg=fixture_path)
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("SL 8.1 checksums", lambda: test_checksums(package, fixture)),
        ("SL 8.2 ONNX layout", lambda: test_onnx_layout(package, sl_cfg)),
    ]

    jags_tests = [
        ("SL 8.3 predict parity", lambda: test_predict_parity(package, sl_cfg)),
        ("SL 8.4 omega1 parity", lambda: test_omega1_parity(package, sl_cfg)),
        (
            "SL 8.5 omega_total parity",
            lambda: test_omega_total_parity(package, sl_cfg, build_dir),
        ),
        (
            "SL 8.6a fixture sanity",
            lambda: test_logdens_fixture_sanity(package, fixture, sl_cfg, build_dir),
        ),
        (
            "SL 8.6b JAGS logdens parity",
            lambda: test_logdens_jags_parity(package, fixture, sl_cfg, build_dir),
        ),
        (
            "SL 8.6c legacy assembly subset",
            lambda: test_logdens_legacy_subset(package, fixture, sl_cfg, build_dir),
        ),
        ("SL 8.7 deviance SL vs legacy", lambda: test_deviance_sl_vs_legacy(package, fixture, sl_cfg)),
        ("SL 8.9 randomSample PPC", lambda: test_random_sample_ppc(package, sl_cfg)),
        (
            "SL 8.11 raw obs logdens parity",
            lambda: test_logdens_raw_obs_parity(package, fixture, sl_cfg, build_dir),
        ),
        ("SL 8.10 dmnorm cross-check", lambda: test_logdens_dmnorm_parity(package, fixture, sl_cfg)),
    ]
    if so_path.exists():
        tests.extend(jags_tests)
    else:
        for name, _ in jags_tests:
            print(f"  [SKIP] {name}: module not compiled")

    passed = 0
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, str(exc)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if ok:
            passed += 1

    return passed, len(tests)
