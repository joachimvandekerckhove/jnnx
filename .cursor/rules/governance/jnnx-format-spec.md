# Neural Network Export Specification (.jnnx Format)

## Overview

The `.jnnx` format is a standardized package for exporting trained neural network models from our cognitive modeling system. Each package contains all necessary files to recreate and deploy a trained neural network emulator.

## Package Structure

```
<model_short_name>.jnnx/
├── model.onnx                    # Trained neural network model
├── scalers.pkl                   # Scaling parameters (Python pickle)
├── scalers.txt                   # Human-readable scaling parameters
├── metadata.json                 # Model specifications and metadata
└── README.md                     # Usage documentation
```

## File Specifications

### 1. `model.onnx`
- **Format**: ONNX (Open Neural Network Exchange)
- **Content**: Trained neural network weights and architecture
- **Platform**: Platform-independent
- **Requirements**: ONNX Runtime (Python or C++)

### 2. `scalers.pkl`
- **Format**: Python pickle file
- **Content**: Scaling parameters for input/output transformation
- **Structure**:
  ```python
  {
      'x_min': [float, ...],      # Minimum values for each input parameter
      'x_max': [float, ...],      # Maximum values for each input parameter  
      'y_min': [float, ...],      # Minimum values for each output
      'y_max': [float, ...]       # Maximum values for each output
  }
  ```

### 3. `scalers.txt`
- **Format**: Plain text, one parameter per line
- **Content**: Human-readable scaling parameters
- **Order**: x_min values, then x_max values, then y_min values, then y_max values
- **Example**:
  ```
  0.5000057380823483
  -4.9998599973887075
  6.96651965448325e-07
  4.999969657432505
  4.999999690904975
  0.999994498339235
  1.4501564355545054e-11
  0.048812367330869924
  0.0010489796813504021
  0.9999999999846692
  7.093237453389978
  25.81461061443715
  ```

### 4. `metadata.json`
- **Format**: JSON
- **Content**: Model specifications and metadata
- **Structure**:
  ```json
  {
      "model_name": "ddm_emulator",
      "module_name": "ddm_emulator",
      "function_name": "ddm_emulator",
      "version": "1.0.0",
      "input_parameters": [
          {"name": "boundary", "min": 0.5, "max": 5.0},
          {"name": "drift", "min": -5.0, "max": 5.0},
          {"name": "ndt", "min": 0.0, "max": 1.0}
      ],
      "output_parameters": [
          {"name": "accuracy", "min": 0.0, "max": 1.0},
          {"name": "mean_rt", "min": 0.0, "max": 10.0},
          {"name": "var_rt", "min": 0.0, "max": 100.0}
      ],
      "transformations": {
          "input_transform": "minmax",
          "output_transforms": ["probit", "log", "log"]
      }
  }
  ```

- **Required Fields**:
  - `model_name` (string): Short identifier for the model
  - `module_name` (string): Name of the compiled JAGS module and artifacts
  - `function_name` (string): Name of the function exposed in JAGS
  - `version` (string): Semantic version of the model package
  - `input_parameters` (array): List of inputs with `name`, `min`, `max`
  - `output_parameters` (array): List of outputs with `name`, `min`, `max`

The `module_name` and `function_name` are used during C++ code generation. They must be present; there are no defaults.

### 5. `README.md`
- **Format**: Markdown
- **Content**: Usage documentation and examples
- **Sections**:
  - Model description
  - Installation requirements
  - Usage examples
  - API reference
  - Troubleshooting

## Usage Examples

### Python Loading
```python
import onnxruntime as ort
import pickle
import json
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Load scaling parameters
with open("scalers.pkl", 'rb') as f:
    scalers = pickle.load(f)

# Load metadata
with open("metadata.json", 'r') as f:
    metadata = json.load(f)

# Transform inputs to [0,1] range
def transform_inputs(inputs):
    x_min = np.array(scalers['x_min'])
    x_max = np.array(scalers['x_max'])
    return (inputs - x_min) / (x_max - x_min)

# Transform outputs back to real-world values
def transform_outputs(outputs):
    y_min = np.array(scalers['y_min'])
    y_max = np.array(scalers['y_max'])
    return outputs * (y_max - y_min) + y_min

# Make prediction
def predict(boundary, drift, ndt):
    inputs = np.array([[boundary, drift, ndt]], dtype=np.float32)
    scaled_inputs = transform_inputs(inputs)
    outputs = session.run(None, {"input": scaled_inputs})[0]
    return transform_outputs(outputs)
```

### C++ Loading
```cpp
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>

class ModelLoader {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<float> x_min, x_max, y_min, y_max;
    
public:
    ModelLoader(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ModelLoader"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        load_scalers();
    }
    
    void load_scalers() {
        // Load scaling parameters from scalers.txt
        std::ifstream file("scalers.txt");
        std::vector<float> params;
        float value;
        while (file >> value) {
            params.push_back(value);
        }
        
        // Split into x_min, x_max, y_min, y_max
        size_t n_inputs = params.size() / 4;
        x_min.assign(params.begin(), params.begin() + n_inputs);
        x_max.assign(params.begin() + n_inputs, params.begin() + 2*n_inputs);
        y_min.assign(params.begin() + 2*n_inputs, params.begin() + 3*n_inputs);
        y_max.assign(params.begin() + 3*n_inputs, params.end());
    }
    
    std::vector<float> predict(float boundary, float drift, float ndt) {
        // Transform inputs
        std::vector<float> inputs = {boundary, drift, ndt};
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = (inputs[i] - x_min[i]) / (x_max[i] - x_min[i]);
        }
        
        // Run inference
        // ... ONNX Runtime inference code ...
        
        // Transform outputs
        // ... output transformation code ...
    }
};
```

## Package Examples

### DDM Model: `ddm.jnnx/`
```
ddm.jnnx/
├── model.onnx
├── scalers.pkl
├── scalers.txt
├── metadata.json
└── README.md
```

### Signal Detection Theory: `sdt.jnnx/`
```
sdt.jnnx/
├── model.onnx
├── scalers.pkl
├── scalers.txt
├── metadata.json
└── README.md
```

### DDM4 Model: `ddm4.jnnx/`
```
ddm4.jnnx/
├── model.onnx
├── scalers.pkl
├── scalers.txt
├── metadata.json
└── README.md
```

## Requirements

### Runtime Dependencies
- **ONNX Runtime**: Version 1.23.0 or compatible
- **Python**: 3.8+ (if using Python interface)
- **C++**: C++17+ (if using C++ interface)

### Platform Support
- **Primary**: Linux x64
- **Secondary**: Windows x64, macOS x64 (with compatible ONNX Runtime)

## Validation

### Package Integrity
- All required files must be present
- `model.onnx` must be valid ONNX format
- `scalers.pkl` must contain required keys
- `metadata.json` must be valid JSON
- `scalers.txt` must have correct number of parameters

### Model Validation
- Input/output dimensions must match metadata
- Scaling parameters must be consistent across files
- Model must produce valid outputs for test inputs
- `metadata.json` must include `module_name` and `function_name`

## Versioning

- **Library version**: Independent of package capabilities (see JNNX release notes)
- **Package capabilities**: Optional `capabilities` array in `metadata.json`; absent means `["emulator"]` only
- **Backward Compatibility**: v1.0 emulator packages without `capabilities` remain valid unchanged

## Package Capabilities

Packages declare what JAGS integrations they support via an optional `capabilities` array in `metadata.json`:

| Capability | JAGS surface | Required sidecars |
|------------|--------------|-------------------|
| `emulator` (default) | `{function_name}` VectorFunction | `scalers.pkl` |
| `synthetic_likelihood` | `{name}_sl` ArrayDist + debug nodes | `likelihood.json` |

Rules:
- `capabilities` omitted → `["emulator"]`
- `capabilities` must always include `emulator`
- Unknown capability names are rejected at validation
- `synthetic_likelihood` capability requires a `synthetic_likelihood` config block and `likelihood.json`

### Emulator-only example (v1.0 compatible)

```json
{
  "model_name": "sdt",
  "module_name": "sdt_emulator",
  "function_name": "sdt_emulator",
  "version": "1.0.0",
  "input_parameters": [...],
  "output_parameters": [...]
}
```

### Synthetic likelihood example

```json
{
  "capabilities": ["emulator", "synthetic_likelihood"],
  "model_name": "ddm3mv",
  "module_name": "ddm3mv_emulator",
  "function_name": "ddm3mv_emulator",
  "version": "1.0.0",
  "input_parameters": [...],
  "output_parameters": [
    {"name": "mu_acc", "group": "mean"},
    {"name": "chol_1", "group": "chol"}
  ],
  "synthetic_likelihood": {
    "n_summaries": 3,
    "onnx_layout": "concatenated",
    "distribution_name": "ddm3mv_sl",
    "trial_count_arg": "n_trials",
    "include_sigma_emu": true,
    "variant": "n_agnostic_cholesky"
  },
  "debug_exports": {
    "predict": true,
    "mean": true,
    "omega1": true,
    "omega_total": true
  }
}
```

### `likelihood.json` (synthetic_likelihood capability)

```json
{
  "version": "1.0",
  "n_summaries": 3,
  "sigma_emu": [[...], [...], [...]]
}
```

ONNX output layout for SL packages: `[mu_std (p), chol_upper (n_chol)]` with `n_chol = p*(p+1)/2`.

`sigma_emu` from `likelihood.json` is baked into the generated C++ module at full double precision (`{val:.17g}`). The codegen `build_manifest.json` records `sigma_emu_baked` and `sl_p` for validation parity (test 8.5).

Deprecated metadata fields: `format_version` and `synthetic_likelihood.enabled` — use `capabilities` instead (see `docs/guides/MIGRATION_v1.1.md`).

Generated JAGS module symbols (single `.so` per package):

- `{function_name}` — emulator alias (mandatory when SL capability present)
- `{model}_predict` — full flat ONNX output
- `{model}_mean`, `{model}_omega1`, `{model}_omega_total` — debug/QA nodes
- `{distribution_name}` — stochastic synthetic likelihood (e.g. `ddm3mv_sl`)

Validate emulator packages with `validate-jnnx` and `validate-module`. SL capability packages additionally run the synthetic-likelihood section of `validate-module`.

## Security Considerations

- **Pickle Files**: Only load from trusted sources
- **ONNX Models**: Validate model integrity before loading
- **Input Validation**: Always validate inputs against parameter ranges

This specification ensures that trained neural network models can be reliably exported and deployed across different facilities while maintaining consistency and usability.
