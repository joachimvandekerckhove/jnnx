# JNNX Installation Guide

## Prerequisites

Before installing JNNX, ensure you have the following dependencies:

### System Requirements

- **Linux** (Ubuntu 18.04+ recommended)
- **JAGS 4.3.0+** - Bayesian analysis software
- **C++ compiler** with C++17 support (g++ 7+ or clang++ 5+)
- **Python 3.8+**

### Installing JAGS

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install jags

# Verify installation
jags --version
```

### Installing ONNX Runtime

```bash
# Download ONNX Runtime (Linux x64)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar -xzf onnxruntime-linux-x64-1.23.2.tgz
sudo mv onnxruntime-linux-x64-1.23.2 /opt/onnxruntime
```

## Installation Methods

### Method 1: pip install from GitHub (Recommended)

```bash
pip install git+https://github.com/joachimvandekerckhove/jnnx.git
```

### Method 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/joachimvandekerckhove/jnnx.git
cd jnnx

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Method 3: Install from Local Directory

```bash
# If you have a local copy
pip install /path/to/jnnx
```

## Verification

After installation, verify that JNNX is working correctly:

```bash
# Check if command-line tools are available
jnnx-setup --help
validate-jnnx --help
generate-module --help
validate-module --help

# Test with included example
validate-jnnx jnnx/models/sdt.jnnx
```

## Python API Test

```python
# Test the Python API
from jnnx import JNNXPackage, validate_jnnx_package

# Load and validate the SDT example
package = JNNXPackage("jnnx/models/sdt.jnnx")
print(f"Package: {package.model_name}")
print(f"Input dim: {package.input_dim}")
print(f"Output dim: {package.output_dim}")

# Validate package
is_valid, errors = validate_jnnx_package("jnnx/models/sdt.jnnx")
print(f"Validation: {'PASS' if is_valid else 'FAIL'}")
```

## Troubleshooting

### Common Issues

#### 1. JAGS Not Found

```
Error: jags: command not found
```

**Solution**: Install JAGS using your package manager or compile from source.

#### 2. ONNX Runtime Not Found

```
Error: Could not load ONNX Runtime
```

**Solution**: Ensure ONNX Runtime is installed and accessible. Check the path in your environment.

#### 3. C++ Compilation Errors

```
Error: fatal error: onnxruntime_cxx_api.h: No such file or directory
```

**Solution**: Install ONNX Runtime headers or adjust include paths in generated Makefiles.

#### 4. Permission Errors

```
Error: Permission denied when installing module
```

**Solution**: Use `sudo` for module installation or adjust JAGS module directory permissions.

### Environment Variables

You can set these environment variables to customize JNNX behavior:

```bash
# ONNX Runtime path
export ONNX_RUNTIME_PATH=/opt/onnxruntime

# JAGS module directory
export JAGS_MODULE_PATH=/usr/lib/x86_64-linux-gnu/JAGS/modules-4

# Build directory
export JNNX_BUILD_DIR=/tmp/jnnx_build
```

### Development Installation

For development work, install in editable mode:

```bash
git clone https://github.com/joachimvandekerckhove/jnnx.git
cd jnnx
pip install -e .
```

This allows you to modify the source code and see changes immediately.

### Uninstallation

To uninstall JNNX:

```bash
pip uninstall jnnx
```

Note: This does not remove JAGS modules that were installed. You may need to manually remove them from the JAGS modules directory.

## Next Steps

After successful installation:

1. **Read the documentation**: See `docs/API.md` for detailed API usage
2. **Try the examples**: Run the notebooks in `demos/` directory
3. **Create your own models**: Follow the `.jnnx` format specification
4. **Generate modules**: Use `generate-module` to create JAGS modules
5. **Integrate with JAGS**: Use `py2jags` to run Bayesian analyses

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review the project documentation
3. Check the GitHub issues page
4. Create a new issue with detailed error information
