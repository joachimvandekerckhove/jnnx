# JNNX - JAGS Neural Network eXtension

JNNX enables evaluation of trained neural networks (ONNX models) as deterministic nodes within JAGS Bayesian models.

## Features

- **ONNX Model Integration**: Load and evaluate ONNX models directly in JAGS
- **Automatic Module Generation**: Generate C++ JAGS modules from ONNX models
- **Scaler Support**: Handle input/output scaling automatically
- **Validation Suite**: Comprehensive testing and validation tools
- **Multiple Model Support**: Works with various neural network architectures

## Installation

```bash
pip install git+https://github.com/joachimvandekerckhove/jnnx.git
```

## Quick Start

1. **Prepare your model**:
   ```bash
   # Create a .jnnx package directory
   mkdir my_model.jnnx
   cp model.onnx my_model.jnnx/
   cp scalers.pkl my_model.jnnx/
   # Create metadata.json (see documentation)
   ```

2. **Generate JAGS module**:
   ```bash
   generate-module my_model.jnnx
   ```

3. **Compile and install**:
   ```bash
   cd tmp/my_model.jnnx_build
   make
   sudo make install
   ```

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

## Requirements

- JAGS 4.3.0+
- ONNX Runtime 1.23.2+
- py2jags 0.1.0+
- Python 3.8+
- C++ compiler with C++17 support

## Documentation

See `docs/` directory for detailed documentation:
- `PROJECT_HANDOFF_MEMO.md`: Complete technical guide
- `jnnx-format-spec.md`: .jnnx package format specification
- `jags-interface-memo.md`: JAGS integration notes

## Examples

See `demos/` directory for example workflows:
- `workflow-sdt.ipynb`: Signal Detection Theory model
- `workflow-ddm.ipynb`: Drift Diffusion Model

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.