# JNNX v1.1.0 feedback (round 2)

**Date:** 2026-07-08  
**From:** ESL / ASL workflow maintainers  
**Re:** JNNX tag `v1.1.0` (`37ed302`)  
**Related:** [jnnx-v1.1-synthetic-likelihood-memo.md](jnnx-v1.1-synthetic-likelihood-memo.md), [jnnx-v1.1-team-response.md](jnnx-v1.1-team-response.md)

---

## 1. Thank you and context

Thank you for shipping v1.1.0. We pulled the tag, generated and compiled `ddm3mv_emulator.so`, and ran `validate-module` against your bundled `models/ddm3mv.jnnx` and our mirrored fixture bundle (`fixtures/ddm3mv_sl_regression.json`, checksums aligned).

The architecture matches what we asked for: a single `.so` with `{name}_emulator`, debug nodes, and `{name}_sl`; one ONNX forward in `logDensity`; and the capabilities model is a clean extension of v1.0. We are ready to migrate our paper-critical models once the items below are resolved.

This memo is detailed on purpose. We want you to have exact reproduction steps, file references, and proposed fixes so we do not go back and forth on symptoms.

---

## 2. Validation summary (our run)

Environment: Linux, JAGS 4.x, ONNX Runtime 1.23.2, py2jags, package at `agent/jnnx` checked out at `v1.1.0`.

| Test | Result | Notes |
|------|--------|-------|
| SL 8.1 checksums | PASS | ONNX and `likelihood.json` SHA256 match fixture |
| SL 8.2 ONNX layout | PASS | `M = 9` for `p = 3` |
| SL 8.3 predict parity | PASS | max diff ~1e-5 |
| SL 8.4 omega1 parity | PASS | max diff ~4e-7 |
| SL 8.5 omega_total parity | **FAIL** | max diff **1.39e-1** (tolerance 1e-3) |
| SL 8.6 logdens parity | PASS | max diff 0.0 over 100 fixture cases |
| SL 8.8 SL smoke | PASS | MCMC with `ddm3mv_sl` completes |

**Important:** 8.6 passing does not imply the compiled JAGS module matches the fixture log-densities. See Section 4.

---

## 3. Issue A: SL 8.5 `_omega_total` parity failure

### 3.1 Symptom

At default test parameters (`theta` = midpoint of input bounds, `n_trials = 600`), Python reference `omega_total_from_chol` disagrees with JAGS `ddm3mv_omega_total(...)` by up to **0.139** on a matrix entry of magnitude ~582 (relative error ~2.4e-4).

`Omega1` at the same `theta` agrees to ~4e-7, so the ONNX path and Cholesky assembly are fine. The divergence appears only after scaling by `N`, inversion, and adding `sigma_emu`.

### 3.2 Root cause (confirmed)

The C++ module bakes `sigma_emu` from `likelihood.json` via codegen in `generate_module.py`. Non-string numeric values are formatted with **six decimal places**:

```102:102:agent/jnnx/jnnx/scripts/generate_module.py
                formatted.append(f"{val:.6f}{suffix}")
```

For `ddm3mv`, this produces constants like:

```29:29:agent/jnnx/tmp/ddm3mv.jnnx_build/ddm3mv_emulator.cc
static const double kSigmaEmu[kP * kP] = {0.000031, -0.000001, -0.000001, ...};
```

whereas `likelihood.json` holds full double precision, e.g. `3.061417492746862e-05`.

SL validation test 8.5 loads **full-precision** `sigma_emu` from `likelihood.json` (`load_sigma_emu` in `scripts/compute_sl_logdens_ref.py`) and compares against JAGS output from the **rounded** baked constants. That mismatch alone reproduces the observed 0.139 discrepancy:

| Comparison | max abs diff on `Omega_total[2,2]` |
|------------|--------------------------------------|
| JAGS vs Python (full `likelihood.json` sigma) | **0.139** |
| JAGS vs Python (codegen-rounded sigma) | **~0** |
| Python full vs Python rounded sigma | **0.138** |

So the C++ math and JAGS wiring appear correct; the test reference and the compiled module disagree on which `sigma_emu` is authoritative.

### 3.3 Secondary note: jitter asymmetry

`omega_total_from_chol` in `sl_math.cpp` optionally adds `1e-10` diagonal jitter before the final inversion (`apply_jitter=true` by default). `mvn_logdens_precision` adds the same jitter again on `Omega_total`. Python references (`sl_reference.py`, `compute_sl_logdens_ref.py`) apply **no** jitter.

This did not cause the 8.5 failure (jitter is negligible next to the sigma rounding effect), but it is a contract inconsistency worth documenting or aligning.

### 3.4 What we need

Please pick one authoritative rule and apply it consistently:

**Option 1 (preferred): full precision in codegen**

- Emit `kSigmaEmu` with sufficient precision (e.g. `{val:.17g}` or hex float literals) so the compiled module matches `likelihood.json` bit-for-bit within reasonable tolerance.
- Update test 8.5 reference to use the same values the module was built with (or compare against values parsed from the generated `.cc` / build manifest).

**Option 2: document rounding and test against baked values**

- State in the format spec that `sigma_emu` is frozen at codegen precision (6 decimals today).
- Change 8.5 (and fixture generation) to round `sigma_emu` the same way before computing the reference.
- Regenerate `fixtures/ddm3mv_sl_regression.json` with rounded sigma so 8.6 tests the module actually shipped.

Either option is fine; inconsistency between `likelihood.json`, codegen, and validation references is the bug.

### 3.5 Suggested acceptance criteria for 8.5

- Absolute tolerance `1e-3` is appropriate for `p = 3` if references use the same `sigma_emu` as the module.
- For larger `p` (we will eventually need up to `p = 17`), consider **relative** tolerance on matrix entries (e.g. `rtol = 1e-6`) since `Omega_total` diagonal entries can be O(10^2) or larger.

---

## 4. Issue B: SL 8.6 does not test JAGS `logDensity` (critical gap)

### 4.1 What the design memo asked for

Section 8.6 of our design memo specifies:

```
logdens_jags = logDensity from {name}_sl  # via py2jags helper
logdens_ref  = mvn_logdens(obs_std, mu_onnx, omega_total_ref(...))
assert abs(logdens_jags - logdens_ref) < 1e-4
```

### 4.2 What v1.1.0 implements

`test_logdens_parity` in `jnnx/sl_validation.py` compares **Python** `sl_logdens(...)` (from `scripts/compute_sl_logdens_ref.py`) to precomputed values in the fixture. It never calls JAGS or inspects `{name}_sl::logDensity`.

Because both fixture and `sl_logdens` use full-precision `sigma_emu` from `likelihood.json`, 8.6 passes with max diff 0.0. The **compiled module** uses rounded `kSigmaEmu`, so JAGS log-densities can differ from the fixture.

We quantified this on all 100 fixture cases using codegen-rounded sigma:

- max |logdens_rounded - logdens_fixture| = **1.23e-3**
- fixture `atol` = **1e-4**

So a proper JAGS-vs-reference test would fail today even though 8.6 reports PASS.

### 4.3 What we need

1. **A real 8.6 test** that evaluates `{name}_sl` log-density through JAGS (or a small C++ test harness that calls the same `logDensity` path). Our team response (2026-07-06) noted py2jags may make direct log-density inspection awkward; if so, a `logDensity` unit test in C++ plus a deviance-equivalence JAGS test would suffice.
2. Reference `sigma_emu` must match baked `kSigmaEmu` (see Issue A).
3. Compare against legacy v1.0 JAGS assembly on a subset of cases (Section 8.6 second bullet in the design memo). This is our ultimate regression guard for migration.

---

## 5. Issue C: posterior predictive checks (PPC) blocked

### 5.1 Current behavior

`SL_Distribution::randomSample` writes `NaN` for all components:

```371:378:agent/jnnx/jnnx/templates/sl_module.cc.template
    void randomSample(double* x, unsigned int length,
                      ...
    {
        for (unsigned int i = 0; i < length; ++i) {
            x[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }
```

That is reasonable if `{name}_sl` is observation-only. It does mean we cannot write:

```jags
obs_rep[1:p] ~ ddm3mv_sl(v, a, t0, n_trials)
```

for posterior predictive simulation.

### 5.2 Our PPC pattern today

VPW08 models (`scripts/ddm3mv/vpw08_models.py`) use a separate replicate likelihood:

```jags
obs_rep[cell,1:p] ~ dmnorm(pred[cell,1:p], Omega_total[cell,1:p,1:p])
```

with `Omega_total` built from the legacy Cholesky assembly (or, after migration, from `{slug}_omega_total(...)`).

### 5.3 What we need

We do not require `randomSample` on `{name}_sl` if the following documented pattern is supported and tested:

```jags
Omega_total[1:p,1:p] <- ddm3mv_omega_total(v, a, t0, n_trials)
obs_rep[1:p] ~ dmnorm(mu, Omega_total[1:p,1:p])   # mu from _mean or pred slice
```

Concretely:

1. Fix Issue A so `_omega_total` is trustworthy at QA tolerance.
2. Add a short "PPC recipe" subsection to `docs/guides/SYNTHETIC_LIKELIHOOD.md`.
3. Optional but valuable: implement `randomSample` on a multivariate Gaussian with precision `Omega_total` (Cholesky of precision or covariance) for a one-line PPC replicate. Low priority if the recipe above is documented.

We will hold PPC re-runs until Issue A is resolved.

---

## 6. Issue D: fixture path rigidity

### 6.1 Current behavior

`_load_fixture` in `sl_validation.py` searches only under the **JNNX repository root**:

```31:35:agent/jnnx/jnnx/sl_validation.py
    candidates = [
        project / "fixtures" / f"{slug}_sl_regression.json",
        project / "fixtures" / "ddm3mv_sl_regression.json",
        project / "docs" / "internal" / "fixtures" / "ddm3mv_sl_regression.json",
    ]
```

Training repos that keep `.jnnx` packages under `models/` and fixtures under `fixtures/` cannot run SL validation in place without copying artifacts into a JNNX checkout.

### 6.2 What we need

Support one or more of:

- `--fixture path/to/{slug}_sl_regression.json` on `validate-module`
- `{slug}_sl_regression.json` adjacent to the package: `models/ddm4mv.jnnx/../fixtures/` or inside the package
- Environment variable `JNNX_FIXTURES_DIR`

We will produce per-slug fixtures (`ddm4mv`, `ddmcollapsesig`, etc.) as we wire models; path flexibility avoids duplicate fixture trees.

---

## 7. Issue E: deviance monitoring with custom `ArrayDist`

### 7.1 Our usage

VPW08 fits monitor `deviance` (via JAGS `dic` module) alongside structural parameters (`scripts/ddm3mv/vpw08_models.py`, `monitor_params()`). Under the legacy workflow, `obs_std ~ dmnorm(...)` is a built-in JAGS distribution; deviance is well defined.

After migration we will use:

```jags
obs_std[cell,1:p] ~ ddm3mv_sl(v[cell], a[cell], nondt, n_trials[cell])
```

### 7.2 Question

Does JAGS accumulate deviance correctly for a custom `ArrayDist` whose `logDensity` delegates to ONNX + matrix algebra? We have not yet run a controlled A/B (legacy `dmnorm` vs `ddm3mv_sl` on identical data and seed) because Issues A and B block confidence in numerical equivalence.

### 7.3 What we need

- Your expectation: should `deviance` match between legacy assembly and `{name}_sl` up to Monte Carlo noise when other model structure is identical?
- If yes, we will add a deviance-equivalence check to our migration gate.
- If no, document the recommended DIC / LOO workflow for SL modules.

---

## 8. Minor metadata and packaging notes

### 8.1 `capabilities` vs `format_version` / `enabled`

v1.1.0 uses:

```json
"capabilities": ["emulator", "synthetic_likelihood"]
```

Our early P0 `metadata.json` used `format_version: "1.1.0"` and `synthetic_likelihood.enabled: true` instead. `validate-jnnx` on our hand-authored package would fail without `capabilities`. Please confirm the canonical v1.1 schema in the format spec (capabilities are fine; we will align our `wire.py` emitter).

### 8.2 `likelihood.json` checksum coupling

Hard failure on ONNX / likelihood SHA256 mismatch (8.1) is exactly what we wanted. When we regenerate `sigma_emu` after retraining, we will update both the sidecar and fixture checksums together.

### 8.3 Emulator alias

Confirmed working: `ddm3mv_emulator` and `ddm3mv_predict` both registered; our codebase calls `{slug}_emulator` everywhere today.

---

## 9. Reproduction commands

From a JNNX v1.1.0 checkout with `ddm3mv_emulator.so` installed:

```bash
export ONNXRUNTIME_DIR=/path/to/onnxruntime
export LTDL_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LTDL_LIBRARY_PATH

cd agent/jnnx
python -m jnnx.scripts.generate_module models/ddm3mv.jnnx /tmp/ddm3mv_build
make -C /tmp/ddm3mv_build && sudo make -C /tmp/ddm3mv_build install

python -m jnnx.scripts.validate_module models/ddm3mv.jnnx /tmp/ddm3mv_build
```

Diagnose Issue A interactively:

```python
# Rounded sigma matches JAGS; full likelihood.json sigma does not.
import numpy as np, onnxruntime as ort, py2jags
from pathlib import Path
from scripts.compute_sl_logdens_ref import load_sigma_emu, omega1_from_chol_upper

pkg = Path("models/ddm3mv.jnnx")
theta = [0.0, 1.25, 0.3]
n_trials = 600
out = ort.InferenceSession(str(pkg / "model.onnx")).run(
    None, {"input": np.array([theta], dtype=np.float32)}
)[0][0]
sigma_full = load_sigma_emu(pkg)
sigma_codegen = np.array([
    0.000031, -0.000001, -0.000001,
    -0.000001, 0.000033, -0.000001,
    -0.000001, -0.000001, 0.000033,
]).reshape(3, 3)

def omega_total(chol, n, sigma):
    o1 = omega1_from_chol_upper(chol, 3)
    return np.linalg.inv(np.linalg.inv(n * o1) + sigma)

# JAGS
chains = py2jags.run_jags(
    model_string="""
    model {
      OmegaTot[1:3,1:3] <- ddm3mv_omega_total(0.0, 1.25, 0.3, 600)
      dummy ~ dnorm(0, 1)
    }""",
    data_dict={"n": 1},
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    monitorparams=["OmegaTot"],
    modules=["ddm3mv_emulator"],
)
jags = chains.get_samples("OmegaTot_2_2")[0]

print("JAGS Omega_22:", jags)
print("ref codegen:", omega_total(out[3:], n_trials, sigma_codegen)[1, 1])
print("ref full:   ", omega_total(out[3:], n_trials, sigma_full)[1, 1])
```

Expected: JAGS matches `ref codegen`, not `ref full`.

---

## 10. Priority and our migration gate

| Priority | Item | Blocks |
|----------|------|--------|
| **P0** | Issue A: align `sigma_emu` codegen and validation reference | `_omega_total` QA, PPC via `dmnorm` replicate |
| **P0** | Issue B: JAGS (or C++) `logDensity` parity test | ESL migration sign-off |
| **P1** | Issue C: PPC recipe (and optional `randomSample`) | VPW08 PPC figures |
| **P1** | Issue D: fixture path flexibility | Multi-model CI in training repo |
| **P2** | Issue E: deviance guidance | DIC tables in paper |

We will **not** patch JNNX locally. We are paused on migration (wire.py, recovery re-runs, paper updates) until P0 items are addressed in an upstream release or you confirm a documented workaround we can rely on.

---

## 11. Artifacts we maintain for you

| Path | Purpose |
|------|---------|
| `models/ddm3mv.jnnx/` | Production v1.1 package (add `capabilities` when we refresh metadata) |
| `fixtures/ddm3mv_sl_regression.json` | 100-case regression fixture (`seed=42`) |
| `scripts/compute_sl_logdens_ref.py` | Reference log-density and fixture generator |
| `docs/internal/` | Mirror for JNNX-only checkouts |

We are happy to regenerate fixtures with whichever `sigma_emu` precision rule you adopt.

---

## 12. Summary

v1.1.0 delivers the right shape: one-line SL likelihood, debug nodes, capabilities, and integrated validation. The blocking problems are **validation/reference consistency**, not the high-level design:

1. **8.5 fails** because tests use full-precision `likelihood.json` while the module uses six-decimal `kSigmaEmu`.
2. **8.6 passes misleadingly** because it never exercises JAGS `logDensity` and shares the same full-precision reference as the fixture.
3. **PPC** needs a documented `dmnorm` replicate using `_omega_total` (after A is fixed) or an implemented `randomSample`.
4. **Fixture paths** should be configurable for multi-repo workflows.
5. **Deviance** behavior with `ArrayDist` needs your guidance before we trust DIC comparisons.

Thank you again for the fast v1.1 turnaround. We are keen to migrate as soon as P0 is closed.
