# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-30

### Changed (breaking)
- **Synthetic likelihood packages require `obs_transform.json`** — JAGS `{name}_sl` and `{name}_logdens` accept **raw physical summary statistics**; column transforms (`identity`, `log1p`, `log`, `sqrt`) + StandardScaler are baked into C++ at codegen
- Regression fixtures use `obs` (raw) instead of `obs_std`; fixture version `2.0` includes `obs_transform_sha256`
- `randomSample` and `typicalValue` return raw physical units via inverse transform

### Added
- `jnnx::sl::obs_raw_to_std` / `obs_std_to_raw` with `JnnxTransform` enum
- Python mirrors in `jnnx.sl_reference`; unit tests in `tests/test_sl_obs_transform.py`
- Validation test **8.11** raw obs logdens parity; **8.1** obs_transform checksum
- Integration test `tests/test_sl_integration_ddm3.py`

### Removed
- v1.x standardized-observation workflow for SL packages (no backward compatibility mode)

## [2.0.0a1] - Unreleased

### Changed (in progress)
- JNNX 2.0: synthetic-likelihood packages accept **raw** physical summary statistics in JAGS; observation transform (column transforms + StandardScaler) baked from `obs_transform.json`

## [1.1.2] - 2026-07-08

### Fixed
- **Critical:** `mvn_logdens_precision` quadratic form now uses `diff' * Omega * diff` (precision), matching JAGS `dmnorm(mu, Omega)` — fixes ESL parameter recovery with `{name}_sl`
- Regenerated `fixtures/ddm3mv_sl_regression.json` log-density column with corrected formula
- **SL 8.6c / 8.7:** compare against SciPy `dmnorm` and JAGS `dmnorm` deviance instead of self-referential Python helper
- **SL 8.10:** new `dmnorm` cross-check gate (`{name}_logdens` vs SciPy on JAGS `mean`/`omega_total` nodes)

### Added
- `scripts/jnnx_sl_logdens_probe.py` standalone regression probe from migration-blocker repro
- SciPy parity unit tests in `tests/test_sl_math_reference.py`

## [1.1.1] - 2026-07-08

### Fixed
- **sigma_emu codegen**: emit full double precision (`{val:.17g}`) so baked `kSigmaEmu` matches `likelihood.json`; record `sigma_emu_baked` in `build_manifest.json`
- **SL 8.5**: validation uses baked sigma from build manifest (fixes 0.139 parity failure)
- **`mvn_logdens_precision`**: Cholesky-based log-determinant (fixes incorrect JAGS log-density / deviance)
- **SL 8.6**: split into fixture sanity (8.6a), JAGS `{name}_logdens` parity (8.6b), and legacy node composition (8.6c)

### Added
- **`randomSample`** on `{name}_sl` for one-line PPC (`obs_rep ~ {name}_sl(...)`); validation test 8.9
- **`{model}_logdens`** scalar debug node for direct JAGS log-density QA (test 8.6b)
- **`jnnx/sl_sigma.py`**: helpers to load baked vs sidecar `sigma_emu`
- Fixture discovery: `--fixture`, `JNNX_FIXTURES_DIR`, `{package_parent}/fixtures/`
- Validation tests **8.7** (deviance SL vs legacy) and **8.9** (randomSample PPC)
- Deprecated-schema warnings for `format_version` and `synthetic_likelihood.enabled`
- Python `sl_reference` jitter contract aligned with C++ (`1e-10`)

### Changed
- `validate-module` SL suite expanded to sections 8.1–8.9
- `docs/guides/SYNTHETIC_LIKELIHOOD.md`: PPC recipe, deviance/DIC guidance, fixture paths

## [1.1.0] - 2026-07-08

### Added
- **Capabilities model**: optional `capabilities` array in `metadata.json` (`emulator` default; `synthetic_likelihood` for SL packages)
- Synthetic likelihood: `{name}_sl` stochastic node, debug nodes (`_predict`, `_mean`, `_omega1`, `_omega_total`), mandatory `{name}_emulator` alias in a single `.so`
- `likelihood.json` sidecar, C++ `sl_math` library, shared `OnnxEngine` in `sl_module.cc.template`
- Integrated SL acceptance tests in `validate-module` (sections 8.1–8.6, 8.8)
- `scripts/compute_sl_logdens_ref.py`, example `models/ddm3mv.jnnx`, fixture `fixtures/ddm3mv_sl_regression.json`
- Demo: `demos/ddm3mv_sl_example.py`
- Docs: [SYNTHETIC_LIKELIHOOD.md](docs/guides/SYNTHETIC_LIKELIHOOD.md), updated format spec and API

### Changed
- `generate-module` selects emulator or SL template based on `synthetic_likelihood` capability
- `validate-module` rewritten: fixed py2jags harness (`monitorparams`, vector outputs), integrated SL suite
- `JNNXPackage.validate()` dispatches per capability

## [1.0.1] - 2026-04-22

### Fixed
- `generate-module` CLI (`jnnx/scripts/generate_module.py`): substitute `{{ONNXRUNTIME_DIR_DEFAULT}}` in the generated Makefile (matches `JAGSModule` behavior); require either `scalers.pkl` or `scalers.json` instead of hard-requiring pickle only; drop unused scaler loading and dead `{{X_MIN}}` / `{{X_MAX}}` / `{{Y_MIN}}` / `{{Y_MAX}}` template replacements (those placeholders are not in the C++ template).

### Changed
- `docs/api/SCALERS_FORMAT.md`: rewritten for the raw I/O contract and Python-only scaler files; removed outdated migration / C++ JSON reader narrative.
- `README.md` and `docs/GETTING_STARTED.md`: link the end-to-end tutorial; clarify scaler file options.

### Added
- `docs/guides/END_TO_END_TUTORIAL.md`: pipeline from training through JAGS, with pointer to `demos/end-to-end-ddm.py`.
- `models/ddm.jnnx/metadata.json`: `description` field clarifying `transformations` vs ONNX runtime behavior.

## [1.0.0] - 2025-10-28

### Added
- Initial release of JNNX (JAGS Neural Network eXtension)
- Complete Python package structure with pip installation support
- Core classes: `JNNXPackage` and `JAGSModule`
- Command-line tools: `jnnx-setup`, `validate-jnnx`, `generate-module`, `validate-module`
- Comprehensive API documentation
- Installation guide and examples
- Support for ONNX model integration with JAGS
- Automatic C++ module generation from ONNX models
- Scaler support for input/output normalization
- Validation suite with 28 comprehensive tests
- Example models: SDT (Signal Detection Theory) and DDM (Drift Diffusion Model)
- Jupyter notebook workflows for model compilation and testing
- Integration with py2jags for Bayesian analysis
- Support for VectorFunction-based JAGS modules
- Global module instance for automatic registration
- Error handling and bounds checking
- MIT License

### Technical Details
- Python 3.8+ support
- JAGS 4.3.0+ compatibility
- ONNX Runtime 1.23.2+ integration
- C++17 compilation support
- Cross-platform Linux support
- Comprehensive test coverage
- Modern Python packaging with pyproject.toml

### Documentation
- Complete API documentation in `docs/api/API.md`
- Installation guide in `docs/guides/INSTALLATION.md`
- Examples and tutorials in `docs/examples/EXAMPLES.md`
- Project handoff memo with technical insights
- JAGS interface memo with common issues and solutions
- Contributing guidelines
- README with quick start guide

### Examples
- SDT model workflow notebook
- DDM model workflow notebook
- Complete integration examples
- Performance optimization examples
- Batch processing examples

## [Unreleased]

### Planned Features
- Support for additional neural network architectures
- Enhanced error reporting and debugging tools
- Performance optimizations for large models
- Additional scaler types beyond MinMaxScaler
- Integration with more Bayesian analysis tools
- Cross-platform support (Windows, macOS)
- Docker containerization support
- CI/CD pipeline improvements
