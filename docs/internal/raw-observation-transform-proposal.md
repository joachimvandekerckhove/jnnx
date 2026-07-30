# Proposal: raw summary statistics in JAGS synthetic-likelihood nodes

**Date:** 2026-07-28  
**From:** ASL / ESL integration team  
**To:** JNNX maintainers  
**Re:** observation-space transform for `{slug}_sl` distributions  
**Status:** proposal (not implemented)

---

## Summary

Amortized synthetic likelihood (ASL) emulators are trained on **standardized** summary vectors: RT-based columns receive `log(1+x)`, then all columns are jointly scaled with a `StandardScaler` fit on training data. At inference, JAGS currently expects those **pre-transformed** values as `obs_std`. Users must load a Python-side `target_transform.pkl` (outside the `.jnnx` package), apply the same transform to observed summaries, and pass the result in the JAGS data dictionary.

We propose extending JNNX so the compiled `{slug}_sl` distribution accepts **raw physical summaries** (accuracy, mean RT, RT variance, etc.) and applies the training-time observation transform inside C++ before evaluating the multivariate normal proxy log density.

This mirrors how `sigma_emu` is already baked into the module from `likelihood.json`: the observation map would be baked from a new package artifact at code generation time.

---

## Problem

### Current inference path

```
trial data  -> aggregate summaries (physical units)
           -> Python TargetTransform.transform()   # log1p + StandardScaler
           -> obs_std in py2jags data dict
           -> obs_std[1:p] ~ {slug}_sl(theta, n_trials)
```

### Pain points

1. **Two artifact sets.** The `.jnnx` package (ONNX + `likelihood.json`) is insufficient for inference; users also need `results/<slug>/target_transform.pkl` from the same training run.

2. **No runtime enforcement.** JAGS does not verify that `obs_std` was standardized correctly. Passing raw summaries runs but yields wrong posteriors.

3. **JAGS-only workflows are awkward.** Any pipeline that fits in JAGS without the ESL Python prep step must reimplement log1p + scaler logic by hand.

4. **Documentation mismatch risk.** `scalers.json` in `.jnnx` documents domains but explicitly states the C++ module does not apply output scaling at runtime; observation scaling is a separate, undocumented contract.

### What already works without this change

- **Parameter inputs** to ONNX are scaled inside the exported graph (`ExportMVModel` bakes `x_mean` / `x_scale`).
- **Emulator outputs** (`mu_std`, Cholesky head) are in standardized summary space.
- **`sigma_emu`** is compiled into the SL module from `likelihood.json`.

Only the **observed summary vector** is still the user's responsibility.

---

## Proposed behavior

### Target user experience

```jags
# obs[cell, 1:3] = raw accuracy, mean RT, var RT (physical units)
obs[cell, 1:3] ~ ddm3mv_sl(drift[cell], bound[cell], nondt[cell], n_trials[cell])
```

JAGS data dictionary contains **raw** summaries only. No Python `TargetTransform` step before `run_jags`.

### Internal flow (inside `{slug}_sl::logDensity`)

1. Read raw observation vector `y_raw` from the stochastic node value `x`.
2. Apply forward observation transform (training-time contract):
   - For RT columns: `z_j = log(1 + y_j)`
   - For proportion columns (accuracy, error rate): `z_j = y_j`
   - Standardize: `s_j = (z_j - mean_j) / scale_j`
3. Run ONNX forward pass at proposed `theta` -> `mu_std`, Cholesky head.
4. Assemble `Omega_total` from Cholesky head, `n_trials`, and baked `sigma_emu` (unchanged).
5. Return `mvn_logdens_precision(s, mu_std, Omega_total)` (unchanged).

Steps 3-5 are identical to v1.1.2; step 2 is new.

### Backward compatibility

Default `observation_space: "standardized"` (current behavior). Packages without an observation-transform artifact continue to expect `obs_std`.

New field `observation_space: "raw"` requires a transform artifact in the package and enables the C++ forward map.

---

## Observation transform definition

Source of truth in ASL training (`asl.data.TargetTransform`):

| Step | Rule |
|------|------|
| Column mask | RT columns: names containing `"rt"` (case-insensitive). Proportion columns: `acc`, `rate`, `prob`, etc. |
| Nonlinear | `log1p` on RT columns only |
| Linear | `StandardScaler` on all columns jointly, fit on training `z_mean` rows from `cov_train.csv` |

Fit statistics (`scaler.mean_`, `scaler.scale_`) are fixed at emulator training time and must match the ONNX / `sigma_emu` training run.

**Inverse transform** (for PPC / `randomSample`):

1. Inverse scaler on standardized sample.
2. `expm1` on RT columns, clamp to valid domain (e.g. RT >= 0).

---

## Package format changes

### New file: `obs_transform.json` (recommended)

Shipped inside `.jnnx/` alongside `likelihood.json`:

```json
{
  "version": "1.0",
  "observation_space": "raw",
  "summary_names": ["acc", "rt_mean", "rt_var"],
  "rt_columns": [false, true, true],
  "scaler_mean": [0.82, 0.45, -1.2],
  "scaler_scale": [0.15, 0.22, 0.88]
}
```

Alternative: nest under `likelihood.json` as `"observation_transform"`. Separate file keeps likelihood math and observation preprocessing distinct.

### `metadata.json` extension

Under `synthetic_likelihood`:

```json
{
  "variant": "n_agnostic_cholesky",
  "n_summaries": 3,
  "summary_names": ["acc", "rt_mean", "rt_var"],
  "observation_space": "raw",
  "distribution_name": "ddm3mv_sl",
  "trial_count_arg": "n_trials",
  "include_sigma_emu": true
}
```

`validate-jnnx` should require `obs_transform.json` when `observation_space == "raw"`, and forbid it (or ignore it) when `observation_space == "standardized"`.

### ASL packaging (`wire-to-jags`)

When building the `.jnnx` directory from a trained emulator:

1. Copy ONNX (already done).
2. Write `likelihood.json` with `sigma_emu` (already done).
3. **New:** serialize `target_transform.pkl` fields into `obs_transform.json`.
4. Record transform hash in `build_manifest.json` (parity with `sigma_emu_baked`).

`target_transform.pkl` can remain in `results/<slug>/` for Python plotting and diagnostics, but JAGS inference would not depend on it when using raw-space packages.

---

## C++ / codegen changes

### `jnnx/cpp/sl_math.cpp`

Add (with tests in `tests/cpp/test_sl_math.cpp`):

```cpp
bool obs_raw_to_std(const double* raw, double* std_out, int p,
                    const bool* rt_mask,
                    const double* mean, const double* scale);

bool obs_std_to_raw(const double* std_in, double* raw_out, int p,
                    const bool* rt_mask,
                    const double* mean, const double* scale);
```

Use `log1p` / `expm1` from `<cmath>`. Return false (caller returns -inf log density) on non-finite or invalid inputs (e.g. RT < -1).

### `sl_module.cc.template`

`generate-module` injects constants: `kRtMask[kP]`, `kObsMean[kP]`, `kObsScale[kP]` (when raw mode enabled).

| Method | Change |
|--------|--------|
| `SL_Distribution::logDensity` | If raw mode: `obs_raw_to_std(x, obs_std)` then existing `sl_logdens_value` |
| `LogdensFunction` | Same |
| `randomSample` | Sample in std space; if raw mode, `obs_std_to_raw` before writing `x` |
| `typicalValue` | If raw mode: inverse-transform `mu_std`; else keep current `mu_std` |

No change to ONNX forward pass or `omega_total_from_chol`.

### `generate_module.py`

- Load `obs_transform.json` when present.
- Pass RT mask and scaler arrays into template substitution (same pattern as `SIGMA_EMU_FLAT`).
- Fail codegen if metadata declares `observation_space: "raw"` but artifact is missing.

---

## Validation and tests

### New regression cases

Extend `{slug}_sl_regression.json` (or add `{slug}_sl_raw_regression.json`):

- Raw `obs` vector + `theta` + `n_trials` + reference `logdens`.
- Reference computed in Python: `TargetTransform.transform(raw)` then existing `sl_logdens` reference path.

### `validate-module` tests

| Test | Purpose |
|------|---------|
| Raw obs logdens parity | C++ `{slug}_logdens` matches Python reference on raw inputs |
| Standardized obs backward compat | Packages without raw mode unchanged |
| PPC raw output | `randomSample` returns physical units (acc in [0,1], RT > 0) when raw mode |
| Invalid raw obs | Returns -inf log density (no silent NaN posteriors) |

### Fixture discovery

Same paths as existing SL fixtures (`--fixture`, `fixtures/`, package-relative).

---

## ESL / consumer impact

### JAGS model strings

No change to distribution syntax; only the **meaning** of the observation node changes when using a raw-space package:

```jags
obs[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)   # obs is raw, not obs_std
```

Optional: recommend renaming the data node from `obs_std` to `obs` in user models for clarity (not required by JNNX).

### Python fit scripts

Can drop `load_target_transform` + `.transform()` before `run_jags` when using raw-space packages. Aggregation code still produces physical summaries (unchanged).

### `build_sl_likelihood_line` (ESL)

Could add `obs_name="obs"` default for raw packages; keep `obs_std` as alias for standardized packages.

---

## Design decisions for maintainers

### 1. Single distribution vs separate symbol

**Recommendation:** one `{slug}_sl` with `observation_space` metadata flag.

- Pros: same JAGS syntax; one `.so` per model.
- Cons: package metadata must be read to know which data convention applies.

Alternative: `{slug}_sl_raw` as a separate `ArrayDist` name.

### 2. Where to implement the transform

**Recommendation:** C++ in the SL module (same as `sigma_emu`).

- Aligns with "JAGS module is self-contained for inference."
- Avoids extra ONNX graphs or user-side JAGS transform nodes.

### 3. Column mask source

Do not infer RT vs proportion from numeric values. Ship explicit `rt_columns` (or derive deterministically from `summary_names` using the same rules as ESL `summary_column_masks()`).

### 4. Versioning

Treat as a **format capability bump** (proposed 1.2.x):

- Old packages: no `obs_transform.json`, standardized obs only.
- New packages: optional or required raw mode per metadata.

Document in `CHANGELOG.md` and migration guide.

---

## Scope estimate

| Component | Effort |
|-----------|--------|
| Package format + `validate-jnnx` | Small |
| ASL `wire.py` serialization | Small |
| `sl_math` forward/inverse + C++ tests | Medium |
| `sl_module.cc.template` + codegen | Medium |
| `sl_validation.py` + fixtures | Medium |
| ESL fit script cleanup | Small (mechanical, many files) |

Suggested milestone order:

1. **ddm3mv** raw logdens parity (no PPC inverse yet).
2. PPC `randomSample` / `typicalValue` inverse transform.
3. Roll out to ddm4mv, ddmcollapsesig, dwmv.

---

## Non-goals

- Changing emulator training or ONNX output layout.
- Replacing `target_transform.pkl` for Python-only workflows (plots, diagnostics).
- Automatic transform discovery from data at inference time (parameters are fixed at training).
- Supporting observation transforms other than log1p-on-RT + global StandardScaler without a format extension.

---

## References (ASL / ESL codebase)

| Item | Location |
|------|----------|
| `TargetTransform` | `asl.data.TargetTransform` |
| Saved at training | `results/<slug>/target_transform.pkl` via `train_mv.py` |
| JAGS SL wiring | `asl.wire.build_jnnx_package` |
| Current SL template | `jnnx/templates/sl_module.cc.template` |
| Example JAGS (standardized obs today) | `obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)` |
| VPW08 prep (Python transform today) | `fit_vpw08.py`: `tf.transform(cells["obs_raw"])` |

---

## Request

We would like JNNX maintainers to:

1. Confirm whether `observation_space: "raw"` fits the v1.1 capability model or warrants a format version bump.
2. Agree on `obs_transform.json` schema (or nested alternative).
3. Prioritize C++ forward transform + logdens parity before PPC inverse.
4. Coordinate tag release so ESL can drop the Python-side `obs_std` prep for JAGS fits when raw packages are installed.

Questions and counter-proposals welcome. We can supply a ddm3mv reference package with `obs_transform.json` and an expanded regression fixture once the schema is settled.

---

**Contact:** ASL integration (companion-repo / ESL wire pipeline)
