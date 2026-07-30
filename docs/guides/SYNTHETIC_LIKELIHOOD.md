# Synthetic Likelihood Capability (v2.0)

## Overview

JNNX packages declare a `synthetic_likelihood` capability alongside `emulator`. One compiled `.so` registers:

- `{function_name}` — emulator alias (e.g. `ddm3mv_emulator`)
- `{model}_predict`, `{model}_mean`, `{model}_omega1`, `{model}_omega_total` — debug/QA nodes
- **`{model}_logdens`** — scalar log-density debug node (raw obs components, theta, n_trials)
- `{distribution_name}` — stochastic synthetic likelihood (e.g. `ddm3mv_sl`)

**v2.0 change:** `{name}_sl` and `{name}_logdens` accept **raw physical summary statistics**. The observation transform from `obs_transform.json` is baked into C++ (column nonlinear transforms + StandardScaler). No Python `TargetTransform` step before `run_jags`.

## Package files (SL)

| File | Purpose |
|------|---------|
| `model.onnx` | Dual-head export `[mu_std, chol_upper]` |
| `likelihood.json` | Debiased `sigma_emu` (standardized summary space) |
| **`obs_transform.json`** | Per-column transforms + scaler (v2.0, required) |
| `metadata.json` | Capabilities, parameters, SL config |

### `obs_transform.json`

```json
{
  "version": "1.0",
  "summary_names": ["acc", "rt_mean", "rt_var"],
  "column_transforms": ["identity", "log1p", "log1p"],
  "scaler_mean": [0.82, 0.389, -2.61],
  "scaler_scale": [0.12, 0.195, 0.71]
}
```

Supported transforms: `identity`, `log1p`, `log`, `sqrt`. Applied per column before `(z - mean) / scale`.

## JAGS model

```jags
obs[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

Data dictionary:

```python
data = {"n_trials": 600, "obs": [acc, rt_mean, rt_var]}  # raw physical units
```

Do not pass `sigma_emu` or scaler parameters — baked at compile time.

## Posterior predictive checks

```jags
obs_rep[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

`randomSample` returns replicates in **raw physical space** (inverse transform applied in C++). Validation test **8.9** checks finite draws and basic domain constraints.

## Precision contract

Internal MVN evaluation uses standardized space after the observation transform. `{name}_logdens` and SciPy cross-checks (test **8.10**) apply the same forward transform before comparing to `dmnorm` precision form.

## Validation tests

| Test | Purpose |
|------|---------|
| 8.1 | ONNX + likelihood + **obs_transform** checksums |
| 8.6a–c | Log-density parity (raw obs) |
| 8.7 | SL logdens vs legacy `dmnorm` deviance |
| 8.9 | PPC `randomSample` in raw space |
| 8.10 | SciPy `dmnorm` cross-check |
| 8.11 | Raw obs logdens parity (explicit gate) |

## Reference package

`models/ddm3mv.jnnx/`, `fixtures/ddm3mv_sl_regression.json`, `demos/ddm3mv_sl_example.py`.
