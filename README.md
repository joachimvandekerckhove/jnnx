# jnnx: Just Another Neural Network Exchange

A toolkit for integrating ONNX-format neural networks into JAGS (Just Another Gibbs Sampler) as deterministic functions.

## Overview

jnnx provides a complete pipeline for converting trained neural networks into JAGS modules, enabling neural network evaluation within Bayesian hierarchical models using MCMC simulation.

## Project Structure

```
jnnx/
├── docs/                    # Documentation
│   ├── api/                # API documentation
│   ├── examples/            # Usage examples
│   ├── guides/              # User guides
│   ├── .memory/            # Project memory and specifications
│   └── SCALERS_FORMAT.md   # Scaler format specification
├── scripts/                 # Command-line tools
│   ├── jnnx-setup          # JSON configuration editor
│   ├── generate-module     # Module code generator
│   ├── validate-jnnx       # JNNX validation
│   ├── validate-module     # Module validation
│   ├── update-scalers.py   # Scaler extraction utility
│   └── extract-scalers.py   # Legacy scaler extraction
├── src/                     # Source code
│   └── templates/          # C++ and Makefile templates
├── models/                  # Neural network models
│   ├── sdt.jnnx/           # Signal Detection Theory example
│   ├── ddm4.jnnx/          # Drift Diffusion Model example
│   └── ddm.jnnx/           # Basic DDM example
├── tests/                   # Test suites
├── examples/                # Usage examples
├── demos/                   # Demonstration scripts
├── tools/                   # Development tools
├── data/                    # Data files
└── tmp/                     # Temporary files (git-ignored)
```

## Quick Start

1. **Setup a model configuration:**
   ```bash
   ./scripts/jnnx-setup models/sdt.jnnx/
   ```

2. **Generate JAGS module:**
   ```bash
   ./scripts/generate-module models/sdt.jnnx/
   ```

3. **Compile and install:**
   ```bash
   cd tmp/sdt.jnnx_build/
   make && sudo make install
   ```

4. **Validate the module:**
   ```bash
   ./scripts/validate-module models/sdt.jnnx/
   ```

## Usage in JAGS

```jags
model {
  result <- sdt(signal, noise, criterion)
  dummy ~ dnorm(0, 1)  # Required for CODA output
}
```

## Requirements

- JAGS 4.x
- ONNX Runtime
- Python 3.x (for tools)
- C++17 compiler

## Documentation

See `docs/` directory for detailed documentation, API references, and usage guides.