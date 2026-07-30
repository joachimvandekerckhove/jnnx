"""Integration test: JAGS MCMC with raw DDM3 observations (v2.0)."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "ddm3mv_sl_regression.json"
PACKAGE = ROOT / "models" / "ddm3mv.jnnx"
BUILD_DIR = ROOT / "tmp" / "ddm3mv.jnnx_build"
MODULE_SO = BUILD_DIR / "ddm3mv_emulator.so"

try:
    import py2jags
except ImportError:
    py2jags = None  # type: ignore


def _module_available() -> bool:
    return MODULE_SO.exists()


@unittest.skipIf(py2jags is None, "py2jags not installed")
@unittest.skipUnless(_module_available(), "compiled ddm3mv module not found")
class TestSlIntegrationDdm3(unittest.TestCase):
    def test_mcmc_on_raw_observations(self):
        data = json.loads(FIXTURE.read_text())
        case = data["cases"][0]
        p = data["n_summaries"]

        model_code = f"""
        model {{
            v ~ dnorm(0, 0.25)
            a ~ dunif(0.5, 2.0)
            t0 ~ dunif(0.15, 0.45)
            obs[1:{p}] ~ ddm3mv_sl(v, a, t0, n_trials)
        }}
        """
        jags_data = {
            "n_trials": case["n_trials"],
            "obs": case["obs"],
        }
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=jags_data,
            nchains=2,
            nsamples=100,
            nadapt=200,
            nburnin=100,
            monitorparams=["v", "a", "t0"],
            modules=["ddm3mv_emulator"],
        )
        self.assertIsNotNone(chains)
        for param, lo, hi in [("v", -2.0, 2.0), ("a", 0.5, 2.0), ("t0", 0.15, 0.45)]:
            samples = chains.get_samples(param)
            self.assertTrue(len(samples) > 0)
            self.assertTrue(all(map(lambda x: lo <= x <= hi, samples)), param)
            self.assertTrue(all(map(lambda x: abs(x) < 1e6, samples)), param)


if __name__ == "__main__":
    unittest.main()
