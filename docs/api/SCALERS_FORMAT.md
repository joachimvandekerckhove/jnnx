# Portable Scaler Storage Format for JNNX

## Overview

This document describes a portable way to store neural network scalers in `.jnnx` directories, avoiding PyTorch's complex serialization issues and dependency problems.

## Problem Statement

The original design stored scalers in `.pth` files, but this approach has several issues:

1. **Dependency Issues**: `.pth` files may contain references to custom modules (e.g., `training.neural_network.get_loss_fn`)
2. **Version Compatibility**: PyTorch serialization format changes between versions
3. **Security Concerns**: Loading `.pth` files can execute arbitrary code
4. **Complexity**: Requires full PyTorch installation and complex error handling

## Solution: Portable JSON Format

### File Structure

Each `.jnnx` directory should contain a `scalers.json` file alongside the existing files:

```
models/example.jnnx/
├── example_emulator.json    # Configuration
├── example_emulator.onnx    # Neural network model
├── example_emulator.pth     # PyTorch checkpoint (optional)
└── scalers.json            # Portable scaler data
```

### JSON Schema

```json
{
  "version": "1.0",
  "input_scaler": {
    "type": "MinMaxScaler",
    "data_min": [0.5, -5.0, 0.0, 0.0],
    "data_max": [5.0, 5.0, 1.0, 1.0],
    "feature_range": [0.0, 1.0]
  },
  "output_scaler": {
    "type": "MinMaxScaler",
    "data_min": [0.0, 0.0, 0.0, 0.0, 0.0],
    "data_max": [1.0, 1.0, 1.0, 1.0, 1.0],
    "feature_range": [0.0, 1.0]
  },
  "metadata": {
    "created_by": "jnnx-tools",
    "created_at": "2025-01-27T14:30:00Z",
    "source_file": "example_emulator.pth",
    "description": "Scalers for example neural network"
  }
}
```

### Field Descriptions

#### Root Level
- `version`: Format version (currently "1.0")
- `input_scaler`: Input scaling parameters
- `output_scaler`: Output scaling parameters  
- `metadata`: Additional information

#### Scaler Objects
- `type`: Scaler type (currently only "MinMaxScaler" supported)
- `data_min`: Array of minimum values for each dimension
- `data_max`: Array of maximum values for each dimension
- `feature_range`: Target range for scaling (typically [0.0, 1.0])

#### Metadata
- `created_by`: Tool that created the file
- `created_at`: ISO timestamp of creation
- `source_file`: Original .pth file (if applicable)
- `description`: Human-readable description

## Usage Examples

### Creating Scalers

```python
# Using the extract-scalers.py utility
python3 extract-scalers.py models/example.jnnx/example_emulator.pth

# Or programmatically
from extract_scalers import extract_scalers_to_json
extract_scalers_to_json("models/example.jnnx/example_emulator.pth")
```

### Loading Scalers

```python
# Using the utility function
from extract_scalers import load_scalers_from_json
scalers = load_scalers_from_json("models/example.jnnx/scalers.json")

# Direct JSON loading
import json
with open("models/example.jnnx/scalers.json", 'r') as f:
    scaler_data = json.load(f)

input_scaler = scaler_data['input_scaler']
x_min = input_scaler['data_min']
x_max = input_scaler['data_max']
```

### C++ Integration

The C++ module can read scalers from the JSON file:

```cpp
// Parse JSON file
std::ifstream file("scalers.json");
json scaler_data;
file >> scaler_data;

// Extract scaling parameters
auto input_scaler = scaler_data["input_scaler"];
std::vector<double> x_min = input_scaler["data_min"];
std::vector<double> x_max = input_scaler["data_max"];
```

## Migration Strategy

### Phase 1: Dual Support
- Update `generate-module` to create both `.txt` and `.json` scaler files
- Update validation scripts to check both formats
- Maintain backward compatibility

### Phase 2: JSON Primary
- Make `scalers.json` the primary format
- Update C++ template to read from JSON
- Deprecate `.txt` format

### Phase 3: JSON Only
- Remove `.txt` scaler support
- Simplify codebase
- Full JSON-based workflow

## Benefits

1. **Portability**: No PyTorch dependencies for scaler loading
2. **Human Readable**: Easy to inspect and debug
3. **Version Agnostic**: No dependency on PyTorch versions
4. **Secure**: No arbitrary code execution
5. **Extensible**: Easy to add new scaler types
6. **Cross-Platform**: Works on any system with JSON support

## Implementation Status

- ✅ `extract-scalers.py` utility created
- ✅ JSON schema defined
- ✅ DDM4 example scalers.json created
- ⚠️ C++ template needs JSON parsing library
- ⚠️ Validation scripts need JSON support
- ⚠️ Migration strategy needs implementation

## Next Steps

1. Add JSON parsing library to C++ template (nlohmann/json)
2. Update validation scripts to use JSON format
3. Test end-to-end workflow with JSON scalers
4. Implement migration strategy
5. Update documentation and examples
