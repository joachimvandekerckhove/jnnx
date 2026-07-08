# Synthetic Likelihood Capability

## Overview

JNNX packages can declare a `synthetic_likelihood` capability alongside the default `emulator` capability. This produces a single JAGS module (one `.so`) that registers:

- `{function_name}` — emulator alias (e.g. `ddm3mv_emulator`)
- `{model}_predict` — full flat ONNX output
- `{model}_mean`, `{model}_omega1`, `{model}_omega_total` — debug/QA nodes
- **`{model}_logdens`** — scalar log-density debug node (obs components, theta, n_trials)
- `{distribution_name}` — stochastic synthetic likelihood (e.g. `ddm3mv_sl`)

Existing emulator-only packages (no `capabilities` field) are unchanged.

## Steps for package authors

1. Export ONNX with concatenated output `[mu_std, chol_upper]` (scaling and softplus diagonals baked in).
2. Add `likelihood.json` with debiased `sigma_emu` matrix in standardized summary space.
3. Update `metadata.json`:
   - `"capabilities": ["emulator", "synthetic_likelihood"]`
   - `"synthetic_likelihood": { "n_summaries": 3, "distribution_name": "...", ... }`
   - Extend `output_parameters` with `group: "mean" | "chol"` entries.
4. Run `validate-jnnx` and `validate-module` on the package (SL tests run automatically).
5. Regenerate and reinstall the JAGS module with `generate-module`.

## JAGS model

**Before (manual likelihood assembly):**

```jags
pred[1:9] <- ddm3mv_emulator(v, a, t0)
# ... Cholesky assembly, sigma_emu in data, dmnorm block ...
obs_std[1:3] ~ dmnorm(pred[1:3], Omega_total[1:3,1:3])
```

**After (synthetic_likelihood capability):**

```jags
obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

Remove `sigma_emu` from the JAGS data dictionary; it is compiled into the module via `likelihood.json` at full double precision (recorded in `build_manifest.json` as `sigma_emu_baked`).

## Posterior predictive checks (PPC)

`{distribution_name}` implements `randomSample`, so you can simulate replicates in one line:

```jags
obs_rep[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

Alternatively, use the debug nodes (same algebra as legacy VPW08 models):

```jags
mu[1:3] <- ddm3mv_mean(v, a, t0)
Omega_total[1:3,1:3] <- ddm3mv_omega_total(v, a, t0, n_trials)
obs_rep[1:3] ~ dmnorm(mu[1:3], Omega_total[1:3,1:3])
```

Validation test **8.9** checks that `randomSample` returns finite draws.

## Precision contract

`{name}_sl` and `{name}_logdens` use JAGS `dmnorm` precision parameterization: `Omega` in `dmnorm(mu, Omega)` is the **precision** matrix, and the log-density quadratic term is `diff' * Omega * diff`.

Validation test **8.10** cross-checks `{name}_logdens` against SciPy `multivariate_normal` with `cov = inv(Omega)` computed from JAGS `{name}_mean` / `{name}_omega_total` nodes. This is the migration gate that catches precision-formula regressions.

## Deviance and DIC

JAGS accumulates deviance for custom `ArrayDist` nodes from `logDensity`. For `{name}_sl`, `-deviance/2` from a legacy `obs_std ~ dmnorm(mu, Omega_total)` model should match `{name}_logdens` on identical data (validation test **8.7**). You can monitor `deviance` with the `dic` module when fitting `obs_std ~ {name}_sl(...)`.

## Numerical jitter

C++ `sl_math` adds `1e-10` diagonal jitter before inverting `Sigma_total` and again when evaluating `mvn_logdens_precision`. Python references in `jnnx/sl_reference.py` mirror this contract for validation parity.

## Fixture discovery

`validate-module` searches for `{slug}_sl_regression.json` under:

- `--fixture PATH` (explicit)
- `$JNNX_FIXTURES_DIR/{slug}_sl_regression.json`
- `fixtures/` at the JNNX repo root
- `{package_parent}/fixtures/` (e.g. `models/../fixtures/`)
- inside the `.jnnx` package directory

## Reference package

See `models/ddm3mv.jnnx/`, `fixtures/ddm3mv_sl_regression.json`, and `demos/ddm3mv_sl_example.py`.
