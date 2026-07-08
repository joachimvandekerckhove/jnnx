# Synthetic Likelihood Capability

## Overview

JNNX packages can declare a `synthetic_likelihood` capability alongside the default `emulator` capability. This produces a single JAGS module (one `.so`) that registers:

- `{function_name}` — emulator alias (e.g. `ddm3mv_emulator`)
- `{model}_predict` — full flat ONNX output
- `{model}_mean`, `{model}_omega1`, `{model}_omega_total` — debug/QA nodes
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

Remove `sigma_emu` from the JAGS data dictionary; it is compiled into the module via `likelihood.json`.

## Reference package

See `models/ddm3mv.jnnx/`, `fixtures/ddm3mv_sl_regression.json`, and `demos/ddm3mv_sl_example.py`.
