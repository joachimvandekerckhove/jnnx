# End-to-end: train → ONNX → `.jnnx` → JAGS

This tutorial bridges “I have a trained network” and “JAGS is calling it as a deterministic function.” It assumes Linux (or similar) with JAGS 4.x, a C++17 compiler, ONNX Runtime C++ SDK, and Python with `torch`, `onnx`, `numpy`, and JNNX installed (`pip install -e .` from the repo root).

## 1. The one rule: raw I/O in ONNX

JAGS passes **physical** parameter values (e.g. drift −3…3). Your ONNX graph’s **external** inputs and outputs must use those same units. If you trained with `StandardScaler` / `MinMaxScaler`, **bake** the transform into the export (preprocessing nodes, fused weights, or a wrapper module) so the saved `model.onnx` already maps raw → raw.

See `docs/api/API.md` and `docs/api/SCALERS_FORMAT.md`.

## 2. Train and export (PyTorch sketch)

Minimal pattern:

1. Define a module whose `forward` takes a batch `[B, N]` in raw space and returns `[B, M]` in raw space.
2. `torch.onnx.export(..., input_names=["input"], output_names=["output"], ...)` with `dummy` shape `[1, N]`.
3. Optional: use `dynamic_axes` for batch — JNNX validation allows dynamic batch if feature dims match metadata.

## 3. Build the `.jnnx` package

Create a directory whose name ends in `.jnnx`:

```text
my_emulator.jnnx/
  metadata.json    # model_name, module_name, function_name, input_parameters, output_parameters
  model.onnx
  scalers.pkl      # or scalers.json — required for package validity; documents training domain
```

`metadata.json` must include `module_name` and `function_name` (JAGS module basename and exposed function name). Each parameter needs plausible `min` / `max` for bounds checks and validation sampling.

## 4. Validate

From the repo root:

```bash
python scripts/validate-jnnx.py my_emulator.jnnx
```

Fix any reported errors (missing files, shape mismatch, raw I/O sanity checks).

## 5. Generate C++ and Makefile

```bash
python scripts/generate-module.py my_emulator.jnnx
```

Artifacts appear under `tmp/my_emulator.jnnx_build/` (module `.cc`, `Makefile`, copied `model.onnx`).

## 6. Compile and install

Point **ONNXRUNTIME_DIR** at the extracted ONNX Runtime **root** (the folder that contains `include/` and `lib/`):

```bash
cd tmp/my_emulator.jnnx_build
export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.23.2
make
sudo make install
```

## 7. Call from JAGS

Load your module (same string as `module_name` in metadata) and use the vector function name from `function_name`. Exact JAGS syntax depends on your model; see `docs/examples/EXAMPLES.md` and the SDT/DDM notebooks under `demos/`.

## Runnable demo in this repo

The script **[`demos/end-to-end-ddm.py`](../../demos/end-to-end-ddm.py)** runs a full pipeline: tiny PyTorch MLP → ONNX → `.jnnx` → validate → `generate-module`. If `ONNXRUNTIME_DIR` is set it attempts `make`; if `py2jags` and JAGS are available it runs a short JAGS check.

```bash
python demos/end-to-end-ddm.py
```

## Related docs

- `docs/GETTING_STARTED.md` — first module walkthrough
- `docs/guides/INSTALLATION.md` — dependencies and paths
- `docs/api/API.md` — `JNNXPackage`, `JAGSModule`, return values of `compile()` / `install()`
