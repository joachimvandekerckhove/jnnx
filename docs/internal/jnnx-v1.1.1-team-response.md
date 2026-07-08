# JNNX v1.1.1 — response to round-2 feedback

**Date:** 2026-07-08  
**Tag:** `v1.1.1`  
**Re:** [jnnx-v1.1-feedback-round2.md](jnnx-v1.1-feedback-round2.md)

## Summary

v1.1.1 addresses all P0/P1 items from your memo and implements `randomSample` (Issue C) as requested.

| Issue | Resolution |
|-------|------------|
| **A — SL 8.5 sigma_emu** | Codegen emits `{val:.17g}`; `build_manifest.json` records `sigma_emu_baked`; test 8.5 uses baked values |
| **B — SL 8.6 JAGS logdens** | `{name}_logdens` debug node + tests 8.6a/b/c; Cholesky fix in `mvn_logdens_precision` |
| **C — PPC / randomSample** | `SL_Distribution::randomSample` implemented via `mvn_sample_precision`; test 8.9 |
| **D — fixture paths** | `--fixture`, `JNNX_FIXTURES_DIR`, `{package_parent}/fixtures/` |
| **E — deviance / DIC** | Test 8.7 (SL vs legacy deviance); documented in `SYNTHETIC_LIKELIHOOD.md` |
| **Minor — schema** | Warnings for `format_version` / `synthetic_likelihood.enabled`; canonical schema is `capabilities` |
| **Minor — jitter** | Python `sl_reference` mirrors C++ `1e-10` jitter for JAGS parity tests |

## Validation

After `generate-module` + `make install` on `models/ddm3mv.jnnx`:

```bash
./jnnx/scripts/validate-module.py models/ddm3mv.jnnx
```

Expected SL section: **8.1–8.9** (15 emulator + 11 SL = 16/16 total when module installed).

## PPC recipes

**One-line (new):**

```jags
obs_rep[1:p] ~ ddm3mv_sl(v, a, t0, n_trials)
```

**Debug-node (legacy-compatible):**

```jags
mu[1:p] <- ddm3mv_mean(v, a, t0)
Omega_total[1:p,1:p] <- ddm3mv_omega_total(v, a, t0, n_trials)
obs_rep[1:p] ~ dmnorm(mu[1:p], Omega_total[1:p,1:p])
```

## Deviance

We expect `deviance` from `obs_std ~ ddm3mv_sl(...)` to match the legacy `dmnorm` assembly on identical data (test 8.7). Monitoring `deviance` with the `dic` module remains valid.

## Migration note

Replace `format_version` / `synthetic_likelihood.enabled` with:

```json
"capabilities": ["emulator", "synthetic_likelihood"]
```

See `docs/guides/MIGRATION_v1.1.md` and `.cursor/rules/governance/jnnx-format-spec.md`.
