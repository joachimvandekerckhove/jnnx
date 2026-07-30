#!/usr/bin/env python3
"""
Example: one-line synthetic likelihood with ddm3mv_sl (JNNX v1.1).

Requires compiled and installed ddm3mv_emulator module (see validate-sl).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import py2jags
except ImportError:
    print("py2jags required for this demo")
    sys.exit(1)


def main() -> None:
    p = 3
    model_code = f"""
    model {{
        v ~ dnorm(0, 0.25)
        a ~ dunif(0.5, 2.0)
        t0 ~ dunif(0.15, 0.45)
        obs[1:{p}] ~ ddm3mv_sl(v, a, t0, n_trials)
    }}
    """
    data = {
        "n_trials": 600,
        "obs": [0.75, 0.5, 0.25],
    }
    chains = py2jags.run_jags(
        model_string=model_code,
        data_dict=data,
        nchains=1,
        nsamples=100,
        nadapt=200,
        nburnin=100,
        monitorparams=["v", "a", "t0"],
        modules=["ddm3mv_emulator"],
    )
    print("Posterior means:")
    for param in ["v", "a", "t0"]:
        if param in chains.parameter_names:
            print(f"  {param}: {chains.get_samples(param).mean():.4f}")


if __name__ == "__main__":
    main()
