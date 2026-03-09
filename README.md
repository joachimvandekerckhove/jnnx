# JNNX - JAGS Neural Network eXchange

JNNX enables evaluation of trained neural networks (ONNX models) as deterministic nodes within JAGS Bayesian models.

## Features

- **ONNX Model Integration**: Load and evaluate ONNX models directly in JAGS (raw I/O; scaling baked into the ONNX graph)
- **Automatic Module Generation**: Generate C++ JAGS modules from `.jnnx` packages
- **Portable scalers**: Support for `scalers.pkl` or `scalers.json`; validation checks raw I/O contract
- **Validation Suite**: Package validation, module generation, and optional compile/install
- **Multiple Model Support**: Fixed or dynamic batch dimensions; various architectures

## Installation

```bash
pip install git+https://github.com/joachimvandekerckhove/jnnx.git
```

## Quick Start

1. **Prepare your model**  
   Your ONNX model must use **raw (original-domain) inputs and outputs**; put any scaling inside the ONNX graph. Create a `.jnnx` package:
   ```bash
   mkdir my_model.jnnx
   cp model.onnx my_model.jnnx/
   # Scalers: either scalers.pkl or scalers.json (see docs/api/SCALERS_FORMAT.md)
   cp scalers.pkl my_model.jnnx/   # or create scalers.json
   # metadata.json must include module_name, function_name, input_parameters, output_parameters
   ```
   Use `jnnx-setup my_model.jnnx` to edit metadata, or see `docs/api/API.md`.

2. **Generate JAGS module**:
   ```bash
   python scripts/generate-module.py my_model.jnnx
   ```

3. **Compile and install**  
   Set `ONNXRUNTIME_DIR` to your ONNX Runtime directory, then build:
   ```bash
   cd tmp/my_model.jnnx_build
   export ONNXRUNTIME_DIR=/path/to/onnxruntime-linux-x64-1.23.2
   make
   sudo make install
   ```
   Or run `make ONNXRUNTIME_DIR=/path/to/...` if you prefer not to export.

4. **Use in JAGS**:
   ```python
   import py2jags
   
   model_string = '''
   model {
       result <- my_function(input1, input2)
       dummy ~ dnorm(0, 1)
   }
   '''
   
   result = py2jags.run_jags(
       model_string=model_string,
       data_dict={'n': 1},
       nchains=1, nsamples=1, nadapt=0, nburnin=0,
       monitorparams=['result'],
       modules=['my_model_emulator']
   )
   ```

## Command Line Tools

- `jnnx-setup`: Configure and edit .jnnx packages
- `validate-jnnx`: Validate .jnnx package integrity
- `generate-module`: Generate C++ JAGS module from .jnnx package
- `validate-module`: Test compiled JAGS module

## Testing

Run tests from the project root:

```bash
# Legacy phase-based suite
python tests/test-suite.py

# Comprehensive coverage suite
python tests/test_suite_full.py -v

# Notebook integration smoke check
python scripts/check-workflow-sdt.py
```

Both suites should pass before opening a pull request.

## Requirements

- JAGS 4.3.0+
- ONNX Runtime 1.23.2+
- py2jags 0.1.0+
- Python 3.8+
- C++ compiler with C++17 support

## Documentation

- `docs/api/API.md`: Public API, scaling contract (raw I/O), and CLI reference
- `docs/api/SCALERS_FORMAT.md`: Portable scaler format (`scalers.json` / `scalers.pkl`)
- `docs/SKILL.md`: AI-agent operating playbook for JNNX tasks
- `docs/examples/EXAMPLES.md`: End-to-end examples and tutorials
- `docs/GETTING_STARTED.md`: First module walkthrough
- `docs/guides/INSTALLATION.md`: Installation and environment setup
- `docs/internal/PROJECT_HANDOFF_MEMO.md`: Technical implementation memo

## Examples

- **`demos/end-to-end-ddm.py`**: Full pipeline (train → export ONNX with raw I/O → create `.jnnx` package → validate → generate module). Optional compile and JAGS check. Run: `python demos/end-to-end-ddm.py`
- **Notebooks**: `workflow-sdt.ipynb`, `workflow-ddm.ipynb` for Signal Detection Theory and Drift Diffusion Model

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.