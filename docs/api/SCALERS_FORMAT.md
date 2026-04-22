# Portable scaler storage for JNNX

## Scaling contract (read this first)

The ONNX model in a `.jnnx` package **must** accept **raw (original-domain) inputs** and return **raw (original-domain) outputs**. All scaling (MinMax, standardization, etc.) must be **inside the ONNX graph**. The generated C++ JAGS module passes JAGS parameter values straight through to ONNX and writes ONNX outputs straight back to JAGS — **it does not read `scalers.pkl`, `scalers.json`, or any scaler file at runtime.**

If you export a model that expects scaled `[0, 1]` inputs while your metadata describes physical ranges (e.g. drift −3…3), JAGS will still pass raw values and you will get **silent wrong answers**. See `docs/api/API.md` for the full contract and validation behavior.

## What scaler files are for

`scalers.pkl` and `scalers.json` are **Python-side metadata**:

- Document the training data domain (min/max per feature) for humans and tools.
- Used by `jnnx` when loading a package (`JNNXPackage.scalers`, `get_scaler_parameters()`), `extract-scalers` / `update-scalers`, and validation heuristics.
- **Not** consumed by the compiled JAGS module.

You can ship either format (or both); `JNNXPackage` tries `scalers.pkl` first, then `scalers.json`.

## Why JSON (optional but recommended for portability)

Pickled sklearn objects can pull in PyTorch/custom code paths and version skew. A portable JSON snapshot avoids that for sharing and CI.

## `scalers.json` schema

Each `.jnnx` directory may contain `scalers.json` alongside `metadata.json` and `model.onnx`:

```
models/example.jnnx/
├── metadata.json
├── model.onnx
└── scalers.json
```

Example:

```json
{
  "version": "1.0",
  "input_scaler": {
    "type": "MinMaxScaler",
    "data_min": [0.5, -5.0, 0.0],
    "data_max": [5.0, 5.0, 1.0],
    "feature_range": [0.0, 1.0]
  },
  "output_scaler": {
    "type": "MinMaxScaler",
    "data_min": [0.0, 0.0, 0.0],
    "data_max": [1.0, 10.0, 100.0],
    "feature_range": [0.0, 1.0]
  },
  "metadata": {
    "created_by": "jnnx-tools",
    "created_at": "2025-01-27T14:30:00Z",
    "description": "Training-domain scaler snapshot"
  }
}
```

### Field notes

- **`input_scaler` / `output_scaler`**: `data_min` and `data_max` arrays align with `input_parameters` / `output_parameters` order in `metadata.json`.
- **`type`**: Documentary; loaders treat MinMax-style `data_min` / `data_max` as the authoritative arrays.
- **`metadata`**: Optional provenance.

When loading JSON, `jnnx.core.JNNXPackage._load_scalers()` maps this into the internal dict shape: `x_min`, `x_max`, `y_min`, `y_max` lists.

## `scalers.pkl` format

Supported shapes:

1. **Flat dict:** `{"x_min": [...], "x_max": [...], "y_min": [...], "y_max": [...]}`
2. **Sklearn-style dict:** `{"x_scaler": MinMaxScaler(...), "y_scaler": MinMaxScaler(...)}` — `get_scaler_parameters()` reads `data_min_` / `data_max_`.

## Creating scalers

Use the project scripts (from repo root):

```bash
python scripts/extract-scalers.py path/to/checkpoint.pth
python scripts/update-scalers.py models/example.jnnx
```

Or build `scalers.json` by hand following the schema above.

## Current implementation status

| Area | Behavior |
|------|----------|
| Load package | `scalers.pkl` preferred, else `scalers.json` |
| C++ JAGS module | No scaler I/O; raw ONNX I/O only |
| Validation | ONNX run with metadata min/max samples to catch obvious raw/scaled mismatches (see `JNNXPackage.validate()`) |

There is **no** planned C++ JSON scaler reader for the default template; scaling belongs in ONNX.
