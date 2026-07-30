"""Tests for SL reference math against fixture and C++ binary."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jnnx.sl_reference import mvn_logdens_precision, omega_total_from_chol
from scripts.compute_sl_logdens_ref import sl_logdens, load_sigma_emu

try:
    import onnxruntime as ort
    from scipy.stats import multivariate_normal
except ImportError:
    ort = None
    multivariate_normal = None


FIXTURE = ROOT / "fixtures" / "ddm3mv_sl_regression.json"
PACKAGE = ROOT / "models" / "ddm3mv.jnnx"
CPP_TEST = ROOT / "tests" / "cpp" / "test_sl_math"


@unittest.skipIf(ort is None, "onnxruntime not installed")
class TestSlMathReference(unittest.TestCase):
    def test_cpp_sl_math_binary(self):
        if not CPP_TEST.exists():
            subprocess.run(["make", "-C", str(ROOT / "tests" / "cpp")], check=True)
        result = subprocess.run([str(CPP_TEST)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_fixture_case0_matches_reference_script(self):
        data = json.loads(FIXTURE.read_text())
        case = data["cases"][0]
        sigma_emu = load_sigma_emu(PACKAGE)
        session = ort.InferenceSession(str(PACKAGE / "model.onnx"))
        logdens = sl_logdens(
            np.array(case["obs_std"]),
            np.array(case["theta"]),
            case["n_trials"],
            session,
            sigma_emu,
            n=data["n_summaries"],
        )
        self.assertLess(abs(logdens - case["logdens"]), data["tolerance"]["atol"])

    @unittest.skipIf(ort is None or multivariate_normal is None, "deps missing")
    def test_mvn_logdens_matches_scipy_on_fixture_case0(self):
        data = json.loads(FIXTURE.read_text())
        case = data["cases"][0]
        sigma_emu = load_sigma_emu(PACKAGE)
        session = ort.InferenceSession(str(PACKAGE / "model.onnx"))
        theta = np.array(case["theta"], dtype=np.float32)
        out = session.run(None, {"input": theta.reshape(1, -1)})[0][0]
        p = data["n_summaries"]
        omega = omega_total_from_chol(out[p:], case["n_trials"], sigma_emu, p)
        mu = out[:p]
        omega_work = np.asarray(omega, dtype=np.float64).copy()
        for i in range(omega_work.shape[0]):
            omega_work[i, i] += 1e-10
        got = mvn_logdens_precision(case["obs_std"], mu, omega, p)
        ref = float(
            multivariate_normal.logpdf(
                case["obs_std"], mean=mu, cov=np.linalg.inv(omega_work)
            )
        )
        self.assertLess(abs(got - ref), data["tolerance"]["atol"])

    def test_precision_quad_differs_from_solve_form(self):
        omega = np.array(
            [
                [4.0, 0.5, 0.0],
                [0.5, 3.0, 0.2],
                [0.0, 0.2, 2.0],
            ],
            dtype=np.float64,
        )
        diff = np.array([0.3, -0.1, 0.2], dtype=np.float64)
        wrong = float(diff @ np.linalg.solve(omega, diff))
        right = float(diff @ omega @ diff)
        self.assertGreater(abs(wrong - right), 1e-3)


if __name__ == "__main__":
    unittest.main()
