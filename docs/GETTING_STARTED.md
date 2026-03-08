# Getting Started with JNNX

This guide walks you through creating your first JNNX-backed JAGS module from an ONNX model.

If you want broader context first, see:
- `docs/guides/INSTALLATION.md`
- `docs/examples/EXAMPLES.md`
- `docs/api/API.md`

## What You Will Build

You will:
1. Create a `.jnnx` package directory.
2. Add model/scaler/metadata files.
3. Validate the package.
4. Generate and compile a JAGS module.
5. Run a minimal JAGS call through `py2jags`.

## Prerequisites

- Python 3.8+
- JAGS 4.3.0+
- ONNX Runtime 1.23.2+
- `py2jags` available in your Python environment
- A trained ONNX model file (for example: `model.onnx`)
- Scaler metadata in `scalers.pkl` (MinMax-style dictionary)

Install JNNX:

```bash
pip install -e .
```

## Step 1: Create a Package Folder

From the repository root:

```bash
mkdir -p my_model.jnnx
cp model.onnx my_model.jnnx/
cp scalers.pkl my_model.jnnx/
```

## Step 2: Add `metadata.json`

Create `my_model.jnnx/metadata.json`:

```json
{
  "model_name": "my_model",
  "module_name": "my_model_emulator",
  "function_name": "my_model_fn",
  "version": "1.0.0",
  "description": "Example JNNX model",
  "input_parameters": [
    { "name": "x1", "min": 0.0, "max": 1.0 },
    { "name": "x2", "min": 0.0, "max": 1.0 }
  ],
  "output_parameters": [
    { "name": "y1", "min": 0.0, "max": 1.0 }
  ]
}
```

Notes:
- `module_name` is the JAGS module identifier.
- `function_name` is the JAGS function you call inside model code.
- Input/output list lengths should match your ONNX model dimensions.

## Step 3: Validate the Package

```bash
validate-jnnx my_model.jnnx
```

For local-repo invocation without installation, you can also run:

```bash
python scripts/validate-jnnx.py my_model.jnnx
```

## Step 4: Generate C++ Module Files

```bash
generate-module my_model.jnnx
```

This creates a build directory like:
- `tmp/my_model.jnnx_build/`

For local-repo invocation:

```bash
python scripts/generate-module.py my_model.jnnx
```

## Step 5: Compile and Install

```bash
cd tmp/my_model.jnnx_build
make
sudo make install
```

## Step 6: Use the Module in JAGS

```python
import py2jags

model_string = """
model {
    out <- my_model_fn(x1, x2)
    dummy ~ dnorm(0, 1)
}
"""

result = py2jags.run_jags(
    model_string=model_string,
    data_dict={"x1": 0.2, "x2": 0.7},
    nchains=1,
    nsamples=1,
    nadapt=0,
    nburnin=0,
    monitorparams=["out"],
    modules=["my_model_emulator"]
)
```

## Common Pitfalls

- `.jnnx` directory name must end with `.jnnx`.
- `metadata.json` must include `module_name` and `function_name`.
- Dimension mismatches between metadata and ONNX model cause runtime failures.
- Missing `model.onnx.data` (for external ONNX data) will break generation.

## Next Steps

- Follow a full workflow notebook in `demos/workflow-sdt.ipynb`.
- Explore advanced usage in `docs/examples/EXAMPLES.md`.
- Review API details in `docs/api/API.md`.