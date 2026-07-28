# JNNX — JAGS Neural Network eXchange

JNNX turns trained neural emulators into JAGS modules. The primary workflow is **synthetic likelihood (SL)**: export a dual-head ONNX model, package it as a `.jnnx` bundle, compile one shared library, and fit standardized summary statistics in JAGS with a single stochastic node — for example `obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)`.

Emulator-only packages (deterministic vector functions) remain supported for backward compatibility.

## Features

- **One-line synthetic likelihood** — `{name}_sl` stochastic node with built-in MVN log-density, debiased `sigma_emu`, and `n_trials` scaling
- **Dual-head ONNX contract** — concatenated output `[μ_std (p), chol_upper (n_chol)]`; scaling and Cholesky diagonals baked into the graph
- **Single `.so` per package** — emulator alias, SL distribution, and optional debug nodes (`_mean`, `_omega_total`, `_logdens`, …)
- **Posterior predictive checks** — `randomSample` on `{name}_sl` for one-line PPC (`obs_rep ~ {name}_sl(...)`)
- **Integrated validation** — `validate-module` runs emulator and SL acceptance tests (checksums, parity, SciPy cross-checks)
- **Raw I/O ONNX** — JAGS passes physical parameter values; all scaling lives inside the ONNX graph

## Requirements

- JAGS 4.3.0+
- ONNX Runtime 1.23.2+ (C++ SDK for compilation)
- py2jags 0.1.0+
- Python 3.8+
- C++ compiler with C++17 support

## Installation

```bash
pip install git+https://github.com/joachimvandekerckhove/jnnx.git
```

For development, clone the repo and install in editable mode:

```bash
git clone https://github.com/joachimvandekerckhove/jnnx.git
cd jnnx
pip install -e .
```

See [Installation guide](docs/guides/INSTALLATION.md) for ONNX Runtime setup and environment variables.

---

## End-to-end: ONNX → `.jnnx` → JAGS

This section walks through the **synthetic likelihood** path from a trained network to a running JAGS model. Reference package: [`models/ddm3mv.jnnx/`](models/ddm3mv.jnnx/).

### Step 1 — Export the ONNX model

Your ONNX graph must use **raw (original-domain) inputs and outputs**. JAGS passes physical values (e.g. drift −2…2); if you trained with `StandardScaler` / `MinMaxScaler`, bake the transforms into the export so the saved `model.onnx` maps raw → raw.

For SL packages, the network is a **dual-head emulator** whose flat output is:

```text
[ μ_std (p summaries), chol_upper (p·(p+1)/2 Cholesky entries) ]
```

Conventions (see [`docs/guides/SYNTHETIC_LIKELIHOOD.md`](docs/guides/SYNTHETIC_LIKELIHOOD.md)):

- **Mean head** — predicted standardized summaries (e.g. accuracy, RT mean, RT variance).
- **Chol head** — upper-triangular Cholesky factor of the per-trial precision Ω₁, with positive diagonals (typically via softplus in the graph).
- **Input tensor** — shape `[1, d]` named `input`; **output tensor** — shape `[1, p + n_chol]` named `output`, where `n_chol = p·(p+1)/2`.

PyTorch export sketch:

```python
import torch

# model: raw [B, d] -> raw [B, p + n_chol]
dummy = torch.randn(1, d)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

Compute **debiased `sigma_emu`** (p×p covariance of emulator error in standardized summary space) from held-out simulation batches. You will store this in `likelihood.json`; it is compiled into the C++ module at full double precision.

### Step 2 — Create the `.jnnx` package

A package is a directory whose name ends in `.jnnx`:

```text
my_model.jnnx/
  metadata.json       # parameters, capabilities, SL config
  model.onnx          # dual-head export from Step 1
  likelihood.json     # sigma_emu (required for SL)
  scalers.pkl         # or scalers.json — documents training domain; used at validation
```

**`metadata.json`** (synthetic likelihood example):

```json
{
  "capabilities": ["emulator", "synthetic_likelihood"],
  "model_name": "ddm3mv",
  "module_name": "ddm3mv_emulator",
  "function_name": "ddm3mv_emulator",
  "version": "1.0.0",
  "input_parameters": [
    {"name": "v",  "min": -2.0,  "max": 2.0},
    {"name": "a",  "min": 0.5,   "max": 2.0},
    {"name": "t0", "min": 0.15,  "max": 0.45}
  ],
  "output_parameters": [
    {"name": "mu_acc",    "group": "mean"},
    {"name": "mu_rt_mean","group": "mean"},
    {"name": "mu_rt_var", "group": "mean"},
    {"name": "chol_1",    "group": "chol"},
    {"name": "chol_2",    "group": "chol"},
    {"name": "chol_3",    "group": "chol"},
    {"name": "chol_4",    "group": "chol"},
    {"name": "chol_5",    "group": "chol"},
    {"name": "chol_6",    "group": "chol"}
  ],
  "synthetic_likelihood": {
    "variant": "n_agnostic_cholesky",
    "n_summaries": 3,
    "summary_names": ["acc", "rt_mean", "rt_var"],
    "onnx_layout": "concatenated",
    "distribution_name": "ddm3mv_sl",
    "trial_count_arg": "n_trials",
    "include_sigma_emu": true
  },
  "debug_exports": {
    "predict": true,
    "mean": true,
    "omega1": true,
    "omega_total": true
  }
}
```

**`likelihood.json`**:

```json
{
  "version": "1.0",
  "n_summaries": 3,
  "sigma_emu": [
    [3.06e-05, -9.82e-07, -9.41e-07],
    [-9.82e-07, 3.26e-05, -9.57e-07],
    [-9.41e-07, -9.57e-07, 3.32e-05]
  ]
}
```

Use `jnnx-setup my_model.jnnx` to create or edit metadata interactively. Field reference: [`docs/api/API.md`](docs/api/API.md), [format spec](.cursor/rules/governance/jnnx-format-spec.md).

### Step 3 — Validate the package

```bash
python scripts/validate-jnnx.py models/ddm3mv.jnnx
```

This checks required files, parameter dimensions, capability declarations, and raw I/O sanity against the ONNX model.

### Step 4 — Generate the JAGS module

```bash
python scripts/generate-module.py models/ddm3mv.jnnx
```

Artifacts land in `tmp/ddm3mv.jnnx_build/` (generated C++, `Makefile`, copied `model.onnx`, `build_manifest.json`). SL packages emit `sl_module.cc` with a shared ONNX engine and registered JAGS nodes.

### Step 5 — Compile and install

Point `ONNXRUNTIME_DIR` at the ONNX Runtime root (the folder containing `include/` and `lib/`):

```bash
cd tmp/ddm3mv.jnnx_build
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.23.2
make
sudo make install
```

Validate the compiled module (emulator + SL tests):

```bash
python scripts/validate-module.py models/ddm3mv.jnnx
```

SL validation covers ONNX checksums, node parity, log-density fixtures, SciPy `dmnorm` cross-checks, PPC sampling, and deviance equivalence. Optional fixture override: `--fixture fixtures/ddm3mv_sl_regression.json`.

### Step 6 — Write the JAGS model

Load the module named in `module_name` (here `ddm3mv_emulator`) and use the stochastic node from `synthetic_likelihood.distribution_name`.

**Primary pattern — one-line SL:**

```jags
model {
  # Priors on model parameters (match metadata input_parameters)
  v  ~ dnorm(0, 0.25)
  a  ~ dunif(0.5, 2.0)
  t0 ~ dunif(0.15, 0.45)

  # Standardized summaries ~ synthetic likelihood
  obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
}
```

**Data dictionary** (example):

```python
data = {
    "n_trials": 600,                    # trial count per cell (passed to SL node)
    "obs_std": [z_acc, z_rt_mean, z_rt_var],  # standardized observed summaries
}
```

Do **not** pass `sigma_emu` in JAGS data — it is baked from `likelihood.json` at compile time.

**Posterior predictive check:**

```jags
obs_rep[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
```

**Debug / QA nodes** (same algebra as legacy manual `dmnorm` assembly):

```jags
mu[1:3] <- ddm3mv_mean(v, a, t0)
Omega_total[1:3,1:3] <- ddm3mv_omega_total(v, a, t0, n_trials)
ld <- ddm3mv_logdens(obs_std[1], obs_std[2], obs_std[3], v, a, t0, n_trials)
```

**py2jags example** ([`demos/ddm3mv_sl_example.py`](demos/ddm3mv_sl_example.py)):

```python
import py2jags

model_code = """
model {
    v ~ dnorm(0, 0.25)
    a ~ dunif(0.5, 2.0)
    t0 ~ dunif(0.15, 0.45)
    obs_std[1:3] ~ ddm3mv_sl(v, a, t0, n_trials)
}
"""
chains = py2jags.run_jags(
    model_string=model_code,
    data_dict={"n_trials": 600, "obs_std": [0.0, 0.0, 0.0]},
    nchains=4,
    nsamples=1000,
    nadapt=500,
    nburnin=500,
    monitorparams=["v", "a", "t0"],
    modules=["ddm3mv_emulator"],
)
```

Set `LTDL_LIBRARY_PATH` to include the ONNX Runtime `lib/` directory if JAGS fails to load the module at runtime.

---

## Exposed JAGS nodes (SL package)

One compiled module registers all of the following. Names derive from `model_name` (slug) and `synthetic_likelihood.distribution_name`.

| JAGS symbol | Type | Purpose |
|-------------|------|---------|
| `{function_name}` | deterministic vector | Emulator alias (μ slice); e.g. `ddm3mv_emulator` |
| `{model}_predict` | deterministic vector | Full ONNX flat output `[μ, chol_upper]` |
| `{model}_mean` | deterministic vector | Standardized summary means (length p) |
| `{model}_omega1` | deterministic matrix | Per-trial precision Ω₁ (p×p) |
| `{model}_omega_total` | deterministic matrix | Total precision Ω_total (p×p); args include `n_trials` |
| `{model}_logdens` | deterministic scalar | Log-density debug node |
| `{distribution_name}` | **stochastic** | Synthetic likelihood; e.g. `ddm3mv_sl` |

Debug nodes (`_predict`, `_mean`, `_omega1`, `_omega_total`) can be disabled via `"debug_exports": { ... false }` in metadata for production builds.

---

## Emulator-only packages

If you only need a deterministic neural network inside JAGS (no SL), omit `capabilities` and `likelihood.json`. The generated module exposes a single vector function:

```jags
result[1:M] <- my_emulator(input1, input2)
```

See [`models/sdt.jnnx/`](models/sdt.jnnx/) and [`demos/end-to-end-ddm.py`](demos/end-to-end-ddm.py) for the emulator-only pipeline.

---

## Command-line tools

| Tool | Purpose |
|------|---------|
| `jnnx-setup` | Create or edit `.jnnx` package metadata |
| `validate-jnnx` | Validate package files and ONNX contract |
| `generate-module` | Generate C++ JAGS module and Makefile |
| `validate-module` | Test compiled module (emulator + SL when declared) |

Python API equivalents: `JNNXPackage`, `JAGSModule` — see [`docs/api/API.md`](docs/api/API.md).

---

## Testing

From the project root:

```bash
python tests/test-suite.py
python tests/test_suite_full.py -v
python scripts/check-workflow-sdt.py
```

Both test suites should pass before opening a pull request.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/guides/SYNTHETIC_LIKELIHOOD.md`](docs/guides/SYNTHETIC_LIKELIHOOD.md) | SL capability details, PPC, deviance/DIC, fixture paths |
| [`docs/guides/END_TO_END_TUTORIAL.md`](docs/guides/END_TO_END_TUTORIAL.md) | Narrative walkthrough with runnable demo |
| [`docs/api/API.md`](docs/api/API.md) | Public API, scaling contract, CLI reference |
| [`docs/api/SCALERS_FORMAT.md`](docs/api/SCALERS_FORMAT.md) | `scalers.pkl` / `scalers.json` format |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | First module walkthrough |
| [`docs/guides/INSTALLATION.md`](docs/guides/INSTALLATION.md) | Dependencies and paths |
| [`docs/examples/EXAMPLES.md`](docs/examples/EXAMPLES.md) | Additional examples |

## Examples in this repo

- **[`models/ddm3mv.jnnx/`](models/ddm3mv.jnnx/)** — reference SL package (p = 3)
- **[`demos/ddm3mv_sl_example.py`](demos/ddm3mv_sl_example.py)** — minimal SL fit with py2jags
- **[`demos/end-to-end-ddm.py`](demos/end-to-end-ddm.py)** — train → ONNX → `.jnnx` → compile (emulator-focused)
- **`fixtures/ddm3mv_sl_regression.json`** — SL log-density regression fixture for `validate-module`

## License

MIT License — see [LICENSE](LICENSE).

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
