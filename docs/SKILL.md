---
name: jnnx-agent-workflow
description: Guides an AI agent to use JNNX for package validation, module generation, compilation, and integration checks. Use when the user asks to build, debug, validate, or document JNNX models, .jnnx packages, or JAGS module workflows.
---

# JNNX Agent Skill

This document teaches an AI agent how to operate effectively in this repository when the task involves JNNX usage, debugging, testing, and documentation updates.

## Mission

Use JNNX to turn `.jnnx` packages (ONNX + metadata + scalers) into JAGS modules, validate behavior, and keep tests/docs consistent.

Priorities:
1. Correctness of package and generated module.
2. Reproducible commands and paths.
3. Fast feedback via tests and integration checks.
4. Documentation updates whenever behavior or commands change.

## When To Apply This Skill

Apply when user asks any of the following:
- build/generate/compile a module
- validate a `.jnnx` package
- fix JNNX CLI failures
- test integration with JAGS/py2jags/notebooks
- clean up scripts/docs around JNNX workflows
- create or debug model packages under `models/*.jnnx`

## Repository Landmarks

- Package code: `jnnx/`
  - `jnnx/core.py`
  - `jnnx/utils.py`
  - `jnnx/scripts/*.py` (canonical CLI implementations)
- Root wrappers for repo-local execution: `scripts/*.py`
- Tests:
  - `tests/test-suite.py` (legacy phase-based + integration)
  - `tests/test_suite_full.py` (comprehensive)
- Demos:
  - `demos/workflow-sdt.ipynb`
  - `demos/workflow-ddm.ipynb`
- Docs:
  - `docs/GETTING_STARTED.md`
  - `docs/guides/INSTALLATION.md`
  - `docs/examples/EXAMPLES.md`
  - `docs/api/API.md`
  - `docs/api/SCALERS_FORMAT.md`

## Mental Model

A valid `.jnnx` package minimally includes:
- `model.onnx` (or ONNX with external data sidecar if needed)
- `metadata.json`
- `scalers.pkl` or `scalers.json`

**ONNX I/O contract:** The ONNX model must accept raw (original-domain) inputs and return raw (original-domain) outputs; all scaling must be inside the ONNX graph. See API.md "Scaling contract".

Critical metadata fields used by generation:
- `model_name`
- `module_name`
- `function_name`
- input/output parameter definitions

High-level flow:
1. Validate package (`validate-jnnx`)
2. Generate C++/Makefile (`generate-module`)
3. Compile/install (`make`, `make install`)
4. Validate runtime wiring (`validate-module` and integration tests)

## Command Strategy

Two valid invocation styles exist:

1) Installed console scripts:
- `jnnx-setup`
- `validate-jnnx`
- `generate-module`
- `validate-module`

2) Repo-local script wrappers (preferred in CI/dev in this repo):
- `python scripts/jnnx-setup.py ...`
- `python scripts/validate-jnnx.py ...`
- `python scripts/generate-module.py ...`
- `python scripts/validate-module.py ...`

Use repo-local wrappers when editing this repository to avoid environment ambiguity.

## Standard Workflows

### Workflow A: Create a New Module From a Model

1. Create package directory:
   - `<name>.jnnx/`
2. Add:
   - `model.onnx`
   - `scalers.pkl`
   - `metadata.json`
3. Validate:
   - `python scripts/validate-jnnx.py <name>.jnnx`
4. Generate:
   - `python scripts/generate-module.py <name>.jnnx`
5. Compile:
   - `cd tmp/<name>.jnnx_build && make`
6. Install (if requested):
   - `sudo make install`
7. Sanity-check with JAGS call (or `validate-module`):
   - `python scripts/validate-module.py <name>.jnnx`

### Workflow B: Debug Generation Failure

Checklist:
- Confirm directory ends with `.jnnx`
- Confirm `metadata.json` is valid JSON
- Confirm `module_name` and `function_name` exist
- Confirm `model.onnx` exists
- Confirm `scalers.pkl` loads
- Re-run `validate-jnnx`
- Re-run `generate-module`
- Inspect generated `tmp/<name>.jnnx_build/*.cc` and `Makefile`

### Workflow C: Integration Verification

Run all:
- `python tests/test_suite_full.py -v`
- `python tests/test-suite.py`
- `python scripts/check-workflow-sdt.py`
- `python jnnx/models/ddm.jnnx/test_package.py` (if DDM path is touched)

If environment-dependent warnings appear (for example sklearn version warnings), report them clearly and distinguish warnings from failures.

## Metadata Contract Guidance

`metadata.json` should be internally consistent:
- number of `input_parameters` matches model input dimensionality
- number of `output_parameters` matches model output dimensionality
- parameter names are stable and descriptive
- min/max bounds are sensible for scaler domain

Naming conventions:
- `module_name`: snake_case and ends in `_emulator` (recommended)
- `function_name`: concise JAGS function name used in model code
- `model_name`: human-readable identifier used in logs/docs

## Scalers Guidance

Expected scaler payload is MinMax-style parameter content used by generation/runtime scaling.

When debugging scaler issues:
- verify `scalers.pkl` exists and is loadable
- verify required keys exist (input/output bounds and ranges)
- ensure dimensions align with metadata and ONNX model

If scaler format changed, also update:
- `docs/api/SCALERS_FORMAT.md`
- relevant tests in both suites

## Testing Protocol

For any code change affecting scripts, core logic, generation, or docs with command changes:
1. Run `tests/test_suite_full.py -v`
2. Run `tests/test-suite.py`
3. Run targeted integration checks if relevant

For docs-only edits, run at least a quick reference check:
- verify moved/renamed paths are updated across repository
- ensure command snippets match current script names

## Documentation Update Rules

When behavior/paths/commands change, update all affected docs in same pass:
- `README.md`
- `docs/GETTING_STARTED.md`
- `docs/guides/INSTALLATION.md`
- `docs/examples/EXAMPLES.md`
- `docs/api/API.md`
- `CONTRIBUTING.md` (if contributor workflow changes)

Do not leave mixed old/new command styles in docs unless intentionally documented as alternatives.

## Failure Triage Playbook

### Error: missing `module_name` or `function_name`
- Fix `metadata.json`
- Re-run validate/generate
- Add or adjust tests to lock regression

### Error: ONNX file missing
- Confirm `model.onnx` path inside `.jnnx`
- If external data used, ensure sidecar file exists and is copied when needed

### Error: validate passes but runtime integration fails
- Separate package integrity from runtime environment issues (JAGS module loading, system libs)
- Report exact phase where failure occurs
- Keep tests strict on deterministic phases; for environment-sensitive phases, assert minimum expected behavior

### Error: docs drift
- Search for old command/path strings and patch all references in one commit

## Communication Format For Agent Responses

When reporting back to the user:
1. What changed (files and rationale)
2. What was verified (exact commands and outcomes)
3. Any residual risks/warnings
4. Optional next step

Keep results concrete. Avoid vague statements like "it should work".

## Done Criteria

Task is complete when all apply:
- requested functionality implemented
- tests/integration checks run at appropriate scope
- docs updated for changed behavior
- no stale command/path references remain

## Quick Command Reference

```bash
# Validate package
python scripts/validate-jnnx.py models/sdt.jnnx

# Generate module sources
python scripts/generate-module.py models/sdt.jnnx

# Validate module wiring
python scripts/validate-module.py models/sdt.jnnx

# Full tests
python tests/test_suite_full.py -v
python tests/test-suite.py

# Notebook integration smoke check
python scripts/check-workflow-sdt.py
```
