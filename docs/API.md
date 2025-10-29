# JNNX API Documentation

## Core Classes

### JNNXPackage

Represents a `.jnnx` package containing ONNX model and metadata.

```python
from jnnx import JNNXPackage

# Load a package
package = JNNXPackage("models/sdt.jnnx")

# Access properties
print(package.model_name)    # "sdt_emulator"
print(package.input_dim)     # 2
print(package.output_dim)    # 2
print(package.metadata)      # Dictionary with package metadata
print(package.scalers)       # Dictionary with scaling parameters

# Get ONNX model path
onnx_path = package.get_onnx_path()

# Validate package
is_valid, errors = package.validate()
```

### JAGSModule

Represents a generated JAGS module.

```python
from jnnx import JAGSModule

# Create module from package
package = JNNXPackage("models/sdt.jnnx")
module = JAGSModule(package, "tmp/sdt.jnnx_build")

# Generate C++ code
module.generate_code()

# Compile module
success = module.compile()

# Install module
success = module.install()
```

## Utility Functions

### Package Validation

```python
from jnnx import validate_jnnx_package

# Validate a package
is_valid, errors = validate_jnnx_package("models/sdt.jnnx")
if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

### Metadata and Scaler Loading

```python
from jnnx import load_metadata, load_scalers

# Load metadata
metadata = load_metadata("models/sdt.jnnx")
print(metadata['model_name'])

# Load scalers
scalers = load_scalers("models/sdt.jnnx")
print(scalers['x_min'])
```

### Package Management

```python
from jnnx.utils import find_jnnx_packages, get_package_info, create_jnnx_package

# Find all packages in a directory
packages = find_jnnx_packages("models/")
print(f"Found {len(packages)} packages")

# Get package information
info = get_package_info("models/sdt.jnnx")
print(f"Package size: {info['size']} bytes")

# Create a new package
create_jnnx_package(
    output_path="new_model.jnnx",
    model_name="my_model",
    onnx_path="model.onnx",
    scalers={'x_min': [0, 0], 'x_max': [1, 1], 'y_min': [0], 'y_max': [1]},
    input_params=[{'name': 'param1', 'min': 0, 'max': 1}],
    output_params=[{'name': 'output1', 'min': 0, 'max': 1}]
)
```

### ONNX Model Testing

```python
from jnnx.utils import test_onnx_model
import numpy as np

# Test ONNX model with sample inputs
test_inputs = [
    np.array([[0.0, 0.0]], dtype=np.float32),
    np.array([[1.0, 1.0]], dtype=np.float32)
]

results = test_onnx_model("models/sdt.jnnx/model.onnx", test_inputs)
print(f"Input shape: {results['input_shape']}")
print(f"Output shape: {results['output_shape']}")
print(f"Test results: {results['test_results']}")
```

## Command Line Interface

The package provides command-line tools accessible via entry points:

### jnnx-setup

Configure and edit `.jnnx` packages.

```bash
jnnx-setup models/sdt.jnnx
```

### validate-jnnx

Validate `.jnnx` package integrity.

```bash
validate-jnnx models/sdt.jnnx
```

### generate-module

Generate C++ JAGS module from `.jnnx` package.

```bash
generate-module models/sdt.jnnx
```

### validate-module

Test compiled JAGS module.

```bash
validate-module models/sdt.jnnx
```

## Integration with JAGS

### Using Generated Modules

Once a module is generated and installed, use it in JAGS:

```python
import py2jags

# Define JAGS model using the generated function
model_string = '''
model {
    result <- sdt(dprime, criterion)
    dummy ~ dnorm(0, 1)
}
'''

from jnnx import JNNXPackage
MODULE_NAME = JNNXPackage('models/sdt.jnnx').metadata['module_name']

# Run JAGS with the custom module
result = py2jags.run_jags(
    model_string=model_string,
    data_dict={'n': 1},
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    monitorparams=['result'],
    modules=[MODULE_NAME]  # Load the custom module
)

# Extract results
hit_rate = result.get_samples('result_1')[0]
fa_rate = result.get_samples('result_2')[0]
```

### Complete Workflow Example

```python
from jnnx import JNNXPackage, JAGSModule
import py2jags
import numpy as np

# 1. Load package
package = JNNXPackage("models/sdt.jnnx")

# 2. Validate package
is_valid, errors = package.validate()
if not is_valid:
    raise ValueError(f"Package validation failed: {errors}")

# 3. Generate and compile module
module = JAGSModule(package, f"tmp/{package.model_name}.jnnx_build")
module.generate_code()
module.compile()
module.install()

# 4. Use in JAGS
model_string = f'''
model {{
    result <- {package.model_name}(0.0, 0.0)
    dummy ~ dnorm(0, 1)
}}
'''

result = py2jags.run_jags(
    model_string=model_string,
    data_dict={'n': 1},
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    monitorparams=['result'],
    modules=[package.metadata['module_name']]
)

# 5. Extract and use results
outputs = [result.get_samples(f'result_{i+1}')[0] for i in range(package.output_dim)]
print(f"Model outputs: {outputs}")
```

## Error Handling

All functions raise appropriate exceptions for error conditions:

- `FileNotFoundError`: When required files are missing
- `ValueError`: When package validation fails
- `RuntimeError`: When ONNX model loading fails

```python
try:
    package = JNNXPackage("nonexistent.jnnx")
except FileNotFoundError as e:
    print(f"Package not found: {e}")

try:
    is_valid, errors = package.validate()
    if not is_valid:
        raise ValueError(f"Validation failed: {errors}")
except ValueError as e:
    print(f"Validation error: {e}")
```

## Advanced Usage

### Custom Scaler Integration

```python
from sklearn.preprocessing import MinMaxScaler
import pickle

# Create custom scalers
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit scalers with your data
x_scaler.fit(training_inputs)
y_scaler.fit(training_outputs)

# Save scalers
scalers = {'x_scaler': x_scaler, 'y_scaler': y_scaler}
with open('custom_scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)
```

### Batch Processing

```python
from jnnx.utils import find_jnnx_packages
from jnnx import validate_jnnx_package

# Validate all packages in a directory
packages = find_jnnx_packages("models/")
for package_path in packages:
    is_valid, errors = validate_jnnx_package(package_path)
    print(f"{package_path}: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
```
