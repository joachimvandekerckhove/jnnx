"""Tests for SL reference math against fixture and C++ binary."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compute_sl_logdens_ref import sl_logdens, load_sigma_emu

try:
    import onnxruntime as ort
except ImportError:
    ort = None


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


if __name__ == "__main__":
    unittest.main()
