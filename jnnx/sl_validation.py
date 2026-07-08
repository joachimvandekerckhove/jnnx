"""Synthetic-likelihood validation suite for validate-module."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from jnnx.capabilities import build_sl_config, has_capability, sha256_file
from jnnx.core import JNNXPackage
from jnnx.sl_reference import omega1_from_chol_upper, omega_total_from_chol

ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return ROOT


def _load_fixture(package_dir: Path) -> Dict[str, Any]:
    slug = package_dir.name.replace(".jnnx", "")
    project = _project_root()
    candidates = [
        project / "fixtures" / f"{slug}_sl_regression.json",
        project / "fixtures" / "ddm3mv_sl_regression.json",
        project / "docs" / "internal" / "fixtures" / "ddm3mv_sl_regression.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(
        f"Regression fixture not found for {slug}; expected fixtures/{slug}_sl_regression.json"
    )


def _run_jags_deterministic(module_name: str, model_code: str, monitor: List[str]) -> Any:
    import py2jags

    return py2jags.run_jags(
        model_string=model_code,
        data_dict={"n": 1},
        nchains=1,
        nsamples=1,
        nadapt=0,
        nburnin=0,
        monitorparams=monitor,
        modules=[module_name],
    )


def test_checksums(package: JNNXPackage, fixture: Dict[str, Any]) -> Tuple[bool, str]:
    onnx_h = sha256_file(package.package_path / "model.onnx")
    like_h = package.get_likelihood_sha256()
    if onnx_h != fixture.get("onnx_sha256"):
        return False, "onnx_sha256 mismatch (stale fixture or package)"
    if like_h != fixture.get("likelihood_sha256"):
        return False, "likelihood_sha256 mismatch (stale sidecar or package)"
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


def test_omega_total_parity(package: JNNXPackage, sl_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    from scripts.compute_sl_logdens_ref import load_sigma_emu

    metadata = package.metadata
    module_name = metadata["module_name"]
    omega_total_name = sl_cfg["omega_total_name"]
    input_params = metadata["input_parameters"]
    theta = [(p["min"] + p["max"]) / 2 for p in input_params]
    theta_str = ", ".join(str(t) for t in theta)
    n_trials = 600
    p = sl_cfg["p"]

    session = ort.InferenceSession(package.get_onnx_path())
    out = session.run(None, {"input": np.array([theta], dtype=np.float32)})[0][0]
    sigma_emu = load_sigma_emu(package.package_path)
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
    if max_diff > 1e-3:
        return False, f"omega_total parity max diff {max_diff:.2e}"
    return True, f"omega_total parity max diff {max_diff:.2e}"


def test_logdens_parity(
    package: JNNXPackage,
    fixture: Dict[str, Any],
    sl_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    from scripts.compute_sl_logdens_ref import load_sigma_emu, sl_logdens

    sigma_emu = load_sigma_emu(package.package_path)
    session = ort.InferenceSession(package.get_onnx_path())
    atol = fixture["tolerance"]["atol"]
    max_diff = 0.0
    for case in fixture["cases"]:
        ref = case["logdens"]
        got = sl_logdens(
            np.array(case["obs_std"]),
            np.array(case["theta"]),
            case["n_trials"],
            session,
            sigma_emu,
            n=sl_cfg["p"],
        )
        diff = abs(got - ref)
        max_diff = max(max_diff, diff)
        if diff > atol:
            return False, f"logdens case mismatch diff={diff:.2e} ref={ref} got={got}"
    return True, f"logdens reference parity max diff {max_diff:.2e} (n={len(fixture['cases'])})"


def test_sl_smoke(package: JNNXPackage, sl_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    import py2jags

    metadata = package.metadata
    module_name = metadata["module_name"]
    dist_name = sl_cfg["distribution_name"]
    p = sl_cfg["p"]

    model_code = f"""
    model {{
        v ~ dnorm(0, 0.25)
        a ~ dunif(0.5, 2.0)
        t0 ~ dunif(0.15, 0.45)
        obs_std[1:{p}] ~ {dist_name}(v, a, t0, n_trials)
    }}
    """
    data = {
        "n_trials": 600,
        "obs_std": [0.0] * p,
    }
    chains = py2jags.run_jags(
        model_string=model_code,
        data_dict=data,
        nchains=1,
        nsamples=5,
        nadapt=100,
        nburnin=50,
        monitorparams=["v", "a", "t0"],
        modules=[module_name],
    )
    if chains is None:
        return False, "SL smoke test returned no chains"
    return True, "SL smoke sample completed"


def run_sl_validation(
    package: JNNXPackage,
    build_dir: Optional[Path] = None,
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

    fixture = _load_fixture(package.package_path)
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("SL 8.1 checksums", lambda: test_checksums(package, fixture)),
        ("SL 8.2 ONNX layout", lambda: test_onnx_layout(package, sl_cfg)),
    ]

    jags_tests = [
        ("SL 8.3 predict parity", lambda: test_predict_parity(package, sl_cfg)),
        ("SL 8.4 omega1 parity", lambda: test_omega1_parity(package, sl_cfg)),
        ("SL 8.5 omega_total parity", lambda: test_omega_total_parity(package, sl_cfg)),
        ("SL 8.6 logdens parity", lambda: test_logdens_parity(package, fixture, sl_cfg)),
        ("SL 8.8 SL smoke", lambda: test_sl_smoke(package, sl_cfg)),
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
