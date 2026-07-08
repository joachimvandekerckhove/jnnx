#!/usr/bin/env python3
"""Regression probe: {name}_logdens must match SciPy/JAGS dmnorm reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import py2jags
    from scipy.stats import multivariate_normal
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        default=ROOT / "models" / "ddm3mv.jnnx",
        help="Path to .jnnx package",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "fixtures" / "ddm3mv_sl_regression.json",
        help="Regression fixture with obs_std rows",
    )
    parser.add_argument("--case-index", type=int, default=0, help="Fixture case index")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for log-density match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text())
    case = fixture["cases"][args.case_index]
    obs = case["obs_std"]
    theta = case["theta"]
    n_trials = case["n_trials"]
    v, a, t0 = theta
    data = {"obs_std": obs, "n_trials": n_trials, "n": 1}

    r = py2jags.run_jags(
        model_string=f"""
        model {{
          v <- {v}; a <- {a}; t0 <- {t0}
          mu[1:3] <- ddm3mv_mean(v, a, t0)
          OmegaTot[1:3,1:3] <- ddm3mv_omega_total(v, a, t0, n_trials)
          dummy ~ dnorm(0, 1) T(0, 0)
        }}""",
        data_dict=data,
        monitorparams=["mu", "OmegaTot"],
        nchains=1,
        nsamples=1,
        nadapt=0,
        nburnin=0,
        modules=["ddm3mv_emulator"],
    )
    mu = np.array([r.get_samples(f"mu_{i}")[0] for i in range(1, 4)])
    omega = np.array(
        [
            [r.get_samples(f"OmegaTot_{i}_{j}")[0] for j in range(1, 4)]
            for i in range(1, 4)
        ]
    )
    omega_work = omega.copy()
    for i in range(omega_work.shape[0]):
        omega_work[i, i] += 1e-10
    ld_jags = float(
        multivariate_normal.logpdf(obs, mean=mu, cov=np.linalg.inv(omega_work))
    )

    r2 = py2jags.run_jags(
        model_string=f"""
        model {{
          v <- {v}; a <- {a}; t0 <- {t0}
          ld <- ddm3mv_logdens(obs_std[1], obs_std[2], obs_std[3], v, a, t0, n_trials)
          dummy ~ dnorm(0, 1) T(0, 0)
        }}""",
        data_dict=data,
        monitorparams=["ld"],
        nchains=1,
        nsamples=1,
        nadapt=0,
        nburnin=0,
        modules=["ddm3mv_emulator"],
    )
    ld_sl = float(r2.get_samples("ld")[0])
    diff = abs(ld_sl - ld_jags)
    print(f"ld_sl={ld_sl:.9f} ld_jags={ld_jags:.9f} diff={diff:.3e}")
    if diff >= args.atol:
        raise SystemExit(
            f"FAIL: |ld_sl - ld_jags| = {diff:.3e} >= atol={args.atol}"
        )
    print("PASS")


if __name__ == "__main__":
    main()
