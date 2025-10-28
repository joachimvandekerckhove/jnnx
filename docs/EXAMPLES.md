# JNNX Examples and Tutorials

## Quick Start Tutorial

This tutorial walks through creating and using a JNNX package with a simple neural network model.

### Step 1: Prepare Your Model

First, you need an ONNX model file. For this example, we'll use the included SDT model:

```python
from jnnx import JNNXPackage

# Load the example SDT package
package = JNNXPackage("jnnx/models/sdt.jnnx")
print(f"Model: {package.model_name}")
print(f"Input dimension: {package.input_dim}")
print(f"Output dimension: {package.output_dim}")
```

### Step 2: Validate the Package

```python
from jnnx import validate_jnnx_package

# Validate package integrity
is_valid, errors = validate_jnnx_package("jnnx/models/sdt.jnnx")
if is_valid:
    print("Package validation: PASSED")
else:
    print("Package validation: FAILED")
    for error in errors:
        print(f"  - {error}")
```

### Step 3: Generate JAGS Module

```bash
# Generate C++ module code
generate-module jnnx/models/sdt.jnnx

# Compile the module
cd tmp/sdt.jnnx_build
make
sudo make install
```

### Step 4: Use in JAGS

```python
import py2jags
import numpy as np

# Define JAGS model using the generated function
model_string = '''
model {
    result <- sdt(dprime, criterion)
    dummy ~ dnorm(0, 1)
}
'''

# Run JAGS with custom module
result = py2jags.run_jags(
    model_string=model_string,
    data_dict={'n': 1},
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    monitorparams=['result'],
    modules=['sdt_emulator']
)

# Extract results
hit_rate = result.get_samples('result_1')[0]
fa_rate = result.get_samples('result_2')[0]
print(f"Hit rate: {hit_rate:.3f}")
print(f"False alarm rate: {fa_rate:.3f}")
```

## Creating Your Own Package

### Example: Custom Neural Network

```python
from jnnx.utils import create_jnnx_package
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# 1. Prepare your ONNX model
# (Assume you have trained a model and saved it as 'my_model.onnx')

# 2. Create scalers
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

# Fit scalers with your training data
training_inputs = np.array([[0, 0], [1, 1], [0.5, 0.5]])
training_outputs = np.array([[0.1], [0.9], [0.5]])

x_scaler.fit(training_inputs)
y_scaler.fit(training_outputs)

# 3. Create scalers dictionary
scalers = {
    'x_scaler': x_scaler,
    'y_scaler': y_scaler
}

# 4. Define input/output parameters
input_params = [
    {'name': 'input1', 'min': 0.0, 'max': 1.0},
    {'name': 'input2', 'min': 0.0, 'max': 1.0}
]

output_params = [
    {'name': 'output1', 'min': 0.0, 'max': 1.0}
]

# 5. Create the package
create_jnnx_package(
    output_path="my_model.jnnx",
    model_name="my_model",
    onnx_path="my_model.onnx",
    scalers=scalers,
    input_params=input_params,
    output_params=output_params
)

print("Package created: my_model.jnnx")
```

## Advanced Examples

### Batch Processing Multiple Models

```python
from jnnx.utils import find_jnnx_packages
from jnnx import validate_jnnx_package, JNNXPackage

# Find all packages in a directory
packages = find_jnnx_packages("models/")

results = {}
for package_path in packages:
    try:
        # Validate package
        is_valid, errors = validate_jnnx_package(package_path)
        
        if is_valid:
            # Load package and get info
            package = JNNXPackage(package_path)
            results[package_path] = {
                'status': 'valid',
                'model_name': package.model_name,
                'input_dim': package.input_dim,
                'output_dim': package.output_dim
            }
        else:
            results[package_path] = {
                'status': 'invalid',
                'errors': errors
            }
    except Exception as e:
        results[package_path] = {
            'status': 'error',
            'error': str(e)
        }

# Print results
for package_path, result in results.items():
    print(f"{package_path}: {result['status']}")
    if result['status'] == 'valid':
        print(f"  Model: {result['model_name']}")
        print(f"  Dimensions: {result['input_dim']} -> {result['output_dim']}")
```

### Custom Module Generation

```python
from jnnx import JNNXPackage, JAGSModule
import subprocess

def generate_and_install_module(package_path, custom_build_dir=None):
    """Generate and install a JAGS module with custom build directory."""
    
    # Load package
    package = JNNXPackage(package_path)
    
    # Use custom build directory if provided
    if custom_build_dir is None:
        build_dir = f"tmp/{package.model_name}.jnnx_build"
    else:
        build_dir = custom_build_dir
    
    # Create module
    module = JAGSModule(package, build_dir)
    
    # Generate code (this would call the actual generation logic)
    print(f"Generating module for {package.model_name}")
    print(f"Build directory: {build_dir}")
    
    # Compile
    success = module.compile()
    if not success:
        raise RuntimeError("Module compilation failed")
    
    # Install
    success = module.install()
    if not success:
        raise RuntimeError("Module installation failed")
    
    print(f"Module {package.model_name}_emulator installed successfully")
    return True

# Use the function
try:
    generate_and_install_module("jnnx/models/sdt.jnnx")
except Exception as e:
    print(f"Error: {e}")
```

### Testing ONNX Models

```python
from jnnx.utils import test_onnx_model
import numpy as np

def test_model_with_multiple_inputs(onnx_path, test_cases):
    """Test an ONNX model with multiple input cases."""
    
    # Prepare test inputs
    test_inputs = []
    for case in test_cases:
        test_inputs.append(np.array([case], dtype=np.float32))
    
    # Test the model
    results = test_onnx_model(onnx_path, test_inputs)
    
    print(f"Model: {onnx_path}")
    print(f"Input shape: {results['input_shape']}")
    print(f"Output shape: {results['output_shape']}")
    print("\nTest results:")
    
    for i, (input_case, output) in enumerate(zip(test_cases, results['test_results'])):
        print(f"  Input {i+1}: {input_case} -> Output: {output[0]}")
    
    return results

# Test the SDT model
test_cases = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]

results = test_model_with_multiple_inputs("jnnx/models/sdt.jnnx/model.onnx", test_cases)
```

## Integration Examples

### Using with PyMC

```python
import pymc as pm
import numpy as np
from jnnx import JNNXPackage
import py2jags

# Load JNNX package
package = JNNXPackage("jnnx/models/sdt.jnnx")

# Define PyMC model
with pm.Model() as model:
    # Define priors
    dprime = pm.Normal('dprime', mu=0, sigma=1)
    criterion = pm.Normal('criterion', mu=0, sigma=1)
    
    # Use JAGS function through custom node
    result = pm.Deterministic('result', 
        pm.CustomDist('jnnx_output', 
                      dprime=dprime, 
                      criterion=criterion,
                      dist=jnnx_distribution))
    
    # Define likelihood
    hit_rate = result[0]
    fa_rate = result[1]
    
    # Observed data
    hits = pm.Binomial('hits', n=100, p=hit_rate, observed=75)
    fas = pm.Binomial('fas', n=100, p=fa_rate, observed=25)

# Sample
trace = pm.sample(1000, tune=1000)
```

### Using with Stan

```stan
data {
    int<lower=0> N;
    vector[N] dprime;
    vector[N] criterion;
    int<lower=0, upper=N> hits[N];
    int<lower=0, upper=N> fas[N];
}

parameters {
    real mu_dprime;
    real mu_criterion;
    real<lower=0> sigma_dprime;
    real<lower=0> sigma_criterion;
}

model {
    // Priors
    mu_dprime ~ normal(0, 1);
    mu_criterion ~ normal(0, 1);
    sigma_dprime ~ exponential(1);
    sigma_criterion ~ exponential(1);
    
    // Likelihood
    for (n in 1:N) {
        dprime[n] ~ normal(mu_dprime, sigma_dprime);
        criterion[n] ~ normal(mu_criterion, sigma_criterion);
        
        // Use JNNX function (would need custom Stan function)
        vector[2] sdt_output = sdt_function(dprime[n], criterion[n]);
        
        hits[n] ~ binomial(N, sdt_output[1]);
        fas[n] ~ binomial(N, sdt_output[2]);
    }
}
```

## Performance Optimization

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
from jnnx import validate_jnnx_package

def validate_package_parallel(package_path):
    """Validate a single package."""
    try:
        is_valid, errors = validate_jnnx_package(package_path)
        return package_path, is_valid, errors
    except Exception as e:
        return package_path, False, [str(e)]

# Validate multiple packages in parallel
packages = find_jnnx_packages("models/")

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(validate_package_parallel, packages))

# Process results
for package_path, is_valid, errors in results:
    print(f"{package_path}: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
```

### Memory Management

```python
import gc
from jnnx import JNNXPackage

def process_packages_with_cleanup(package_paths):
    """Process packages with memory cleanup."""
    results = []
    
    for package_path in package_paths:
        try:
            # Load package
            package = JNNXPackage(package_path)
            
            # Process package
            result = {
                'path': package_path,
                'model_name': package.model_name,
                'input_dim': package.input_dim,
                'output_dim': package.output_dim
            }
            results.append(result)
            
        except Exception as e:
            results.append({
                'path': package_path,
                'error': str(e)
            })
        
        finally:
            # Clean up memory
            gc.collect()
    
    return results
```
