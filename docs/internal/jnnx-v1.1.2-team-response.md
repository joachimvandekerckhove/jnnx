# JNNX v1.1.2 — migration blocker resolution

**Date:** 2026-07-08  
**Tag:** `v1.1.2`  
**Re:** [jnnx-v1.1.1-migration-blocker.md](jnnx-v1.1.1-migration-blocker.md)

## Summary

v1.1.2 fixes the precision-matrix bug in `mvn_logdens_precision` that caused `{slug}_sl` log-densities to disagree with JAGS `dmnorm` and broke parameter recovery.

| Item | Resolution |
|------|------------|
| Quadratic form | `diff' * Omega * diff` (was `diff' * Omega^{-1} * diff`) |
| Fixture | `fixtures/ddm3mv_sl_regression.json` logdens column regenerated |
| Validation | **8.10** SciPy `dmnorm` cross-check; **8.6c/8.7** fixed to compare against real `dmnorm` reference |
| Probe | `scripts/jnnx_sl_logdens_probe.py` |

## ESL action

1. Check out `v1.1.2`
2. Regenerate and reinstall `{slug}_emulator.so`
3. Remove `ESL_LEGACY_LIKELIHOOD=1` workaround
4. Re-run recovery / VPW08 refits

## Verification

```bash
python scripts/generate-module.py models/ddm3mv.jnnx
cd tmp/ddm3mv.jnnx_build && make && sudo make install
python jnnx/scripts/validate_module.py models/ddm3mv.jnnx --build-dir tmp/ddm3mv.jnnx_build
python scripts/jnnx_sl_logdens_probe.py
```

Expected: **18/18** validate-module (6 emulator + 12 SL), probe PASS.

## Note on v1.1.1

v1.1.1 fixed log-determinant computation but left the quadratic form wrong. Do not use v1.1.1 for `{name}_sl` production fits.
