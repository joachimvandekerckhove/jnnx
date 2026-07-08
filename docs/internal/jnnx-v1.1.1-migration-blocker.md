# JNNX v1.1.1 migration blocker: root cause and fix

**Date:** 2026-07-08  
**Re:** ESL migration to `{slug}_sl` stochastic nodes (JNNX v1.1.1)  
**Status:** Blocks recovery re-runs and paper number updates  
**JNNX tag tested:** v1.1.1 (`c825f2c`)

---

## Summary

JNNX v1.1.1 passes the integrated `validate-module` SL suite (11/11 for `ddm3mv`). ESL wiring is complete. **Parameter recovery with `obs_std ~ ddm3mv_sl(...)` is wrong** on identical simulated data; legacy `dmnorm` recovery is correct.

**Root cause (confirmed):** `jnnx::sl::mvn_logdens_precision` (C++ and Python mirror) evaluates the MVN quadratic form as `diff' * Omega^{-1} * diff` instead of `diff' * Omega * diff` for a **precision** matrix `Omega`. JAGS `dmnorm(mu, Omega)` uses the correct precision parameterization. Legacy ESL models call JAGS `dmnorm`; `{slug}_sl` and `{slug}_logdens` call the buggy helper. The SL validation suite does not compare against JAGS `dmnorm`, so the bug is invisible to tests 8.6b-8.7.

---

## Root cause detail

### Bug location

`jnnx/cpp/sl_math.cpp`, function `mvn_logdens_precision`:

```cpp
// Current (wrong for precision Omega):
chol_logdet_and_solve(omega_work, p, diff, logdet, sol);  // sol = Omega^{-1} * diff
quad = diff' * sol;  // = diff' * Omega^{-1} * diff

// Correct for JAGS dmnorm(mu, Omega) with Omega = precision:
quad = diff' * Omega * diff;
```

Same bug in `jnnx/sl_reference.py` (`quad = diff @ np.linalg.solve(omega_work, diff)`).

`SL_Distribution::logDensity` and `LogdensFunction` both call `sl_logdens_value` -> `mvn_logdens_precision`, so the stochastic node and debug node are **internally consistent but both wrong relative to JAGS**.

`mvn_sample_precision` inverts `Omega` to `Sigma` before sampling; that path is consistent with a correct likelihood and is not the source of the recovery failure.

### Numeric proof (ddm3mv, fixture case 0)

Fixed `theta = [-0.5, 1.2, 0.25]`, `n_trials = 500`, `obs_std` from `fixtures/ddm3mv_sl_regression.json`:

| Quantity | log-density |
|----------|-------------|
| JAGS `dmnorm(mu, Omega_total)` | 4.554935 |
| SciPy `multivariate_normal` (cov = Omega^{-1}) | 4.554934 |
| Correct formula: `diff' * Omega * diff` | quad = 1.573 |
| **JNNX `ddm3mv_logdens` / `ddm3mv_sl`** | **5.341180** |
| Buggy formula: `diff' * Omega^{-1} * diff` | quad = 7.9e-05 -> ld = 5.341 |

**Log-likelihood bias:** +0.786 nats on this case (SL overweights the likelihood). Across 20 fixture cases: mean bias +1.20 nats, max +2.61 nats.

Deviance check (`dic` module, fixed theta):

| Model | deviance |
|-------|----------|
| `obs_std ~ dmnorm(pred, Omega_total)` (legacy) | -9.110 |
| `obs_std ~ ddm3mv_sl(v, a, t0, n_trials)` | -10.682 |

---

## Why validate-module passes but recovery fails

| Test | What it actually checks | Why it misses the bug |
|------|-------------------------|------------------------|
| 8.6b | JAGS `{name}_logdens` vs Python `mvn_logdens_precision` | Same wrong formula both sides |
| 8.6c | `{name}_logdens` vs Python on JAGS `mean`/`omega_total` nodes | Same wrong formula both sides |
| **8.7** | "integrated vs legacy" | **Both sides evaluate `{name}_logdens`**; `mean`/`omega_total` nodes are unused for the comparison |
| 8.8 smoke | SL node samples without error | Only checks that MCMC runs |
| 8.9 PPC | `randomSample` finite | Sampling path uses covariance correctly |

Legacy recovery uses JAGS `dmnorm` (correct). SL recovery uses `{slug}_sl::logDensity` (buggy). Tests never cross-check SL against `dmnorm`.

---

## Recovery impact (ddm3mv, same subject, same `obs_std`, `n_trials=500`, seed 42)

| Parameter | True | Legacy post. mean | SL post. mean |
|-----------|------|-------------------|---------------|
| v | -0.50 | -0.505 | -0.002 |
| a | 1.20 | 1.240 | 0.754 |
| t0 | 0.25 | 0.229 | 0.303 |

Smoke recovery (30-50 subjects, `ESL_SMOKE=1`):

| Path | corr(v) | corr(a) | corr(t0) | a coverage |
|------|---------|---------|----------|------------|
| Legacy | 0.994 | 0.995 | 0.977 | 0.98 |
| SL | 0.378 | 0.362 | 0.207 | 0.73 |

---

## Suggested fix (JNNX)

In `mvn_logdens_precision`, keep `chol_logdet_and_solve` (or `slogdet`) for `log|Omega|`, but replace the quadratic form:

```cpp
// quad = diff' * Omega * diff  (Omega is precision)
double quad = 0.0;
for (int i = 0; i < p; ++i) {
    double row = 0.0;
    for (int j = 0; j < p; ++j) {
        row += omega_work[static_cast<size_t>(i * p + j)] * diff[static_cast<size_t>(j)];
    }
    quad += diff[static_cast<size_t>(i)] * row;
}
```

Mirror the same change in `jnnx/sl_reference.py`.

### Suggested new tests

1. **8.6d / 8.10:** At fixed `(theta, obs_std, n_trials)`, compare log-density from:
   - `{name}_logdens` (or `{name}_sl` deviance)
   - JAGS `dmnorm({name}_mean(theta), {name}_omega_total(theta, n_trials))`  
   Tolerance: match fixture `atol` (~1e-5).

2. **8.11:** MCMC recovery on 1-3 fixed cases: `{name}_sl` vs legacy `dmnorm` assembly; posterior means should agree within tolerance.

3. **Fix 8.7:** Legacy arm should evaluate `dmnorm` log-density (or deviance), not `{name}_logdens` again.

---

## Minimal reproduction

```bash
cd /path/to/esl
export ONNXRUNTIME_DIR=/path/to/onnxruntime
export LTDL_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LTDL_LIBRARY_PATH

.venv/bin/python scripts/jnnx_sl_logdens_bug_probe.py
```

Or inline (requires wired `ddm3mv_emulator` module):

```python
from py2jags import run_jags
from scipy.stats import multivariate_normal
import numpy as np

obs = [-0.5542791, -0.17957759, -0.10555756]
v, a, t0, n = -0.5, 1.2, 0.25, 500
data = {"obs_std": obs, "n_trials": n, "n": 1}

# mu, Omega from JNNX nodes
r = run_jags(
    model_string=f"""
    model {{
      v <- {v}; a <- {a}; t0 <- {t0}
      mu[1:3] <- ddm3mv_mean(v, a, t0)
      OmegaTot[1:3,1:3] <- ddm3mv_omega_total(v, a, t0, n_trials)
      dummy ~ dnorm(0, 1) T(0, 0)
    }}""",
    data_dict=data, monitorparams=["mu", "OmegaTot"],
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    modules=["ddm3mv_emulator"],
)
mu = np.array([r.get_samples(f"mu_{i}")[0] for i in range(1, 4)])
omega = np.array([[r.get_samples(f"OmegaTot_{i}_{j}")[0] for j in range(1, 4)] for i in range(1, 4)])

ld_jags = multivariate_normal.logpdf(obs, mean=mu, cov=np.linalg.inv(omega))

r2 = run_jags(
    model_string=f"""
    model {{
      v <- {v}; a <- {a}; t0 <- {t0}
      ld <- ddm3mv_logdens(obs_std[1], obs_std[2], obs_std[3], v, a, t0, n_trials)
      dummy ~ dnorm(0, 1) T(0, 0)
    }}""",
    data_dict=data, monitorparams=["ld"],
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    modules=["ddm3mv_emulator"],
)
ld_sl = float(r2.get_samples("ld")[0])
assert abs(ld_sl - ld_jags) < 1e-4, f"SL={ld_sl} JAGS={ld_jags}"  # fails on v1.1.1
```

---

## ESL status

- Code migration: wire.py v1.1, SL model strings, VPW08 builders, joint collapse model.
- **Recovery re-runs, calibration A/B, VPW08 refits, and paper numeric tables are paused.**
- Workaround: `ESL_LEGACY_LIKELIHOOD=1` restores correct recovery via JAGS `dmnorm`.
- No local JNNX patches; awaiting upstream fix.

---

## References

- Bug: `jnnx/cpp/sl_math.cpp` `mvn_logdens_precision` (~lines 218-239)
- Callers: `jnnx/templates/sl_module.cc.template` `sl_logdens_value`, `SL_Distribution::logDensity`
- Misleading test: `jnnx/sl_validation.py` `test_deviance_sl_vs_legacy` (8.7)
