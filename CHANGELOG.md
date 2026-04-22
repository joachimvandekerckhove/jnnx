# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-04-22

### Fixed
- `generate-module` CLI (`jnnx/scripts/generate_module.py`): substitute `{{ONNXRUNTIME_DIR_DEFAULT}}` in the generated Makefile (matches `JAGSModule` behavior); require either `scalers.pkl` or `scalers.json` instead of hard-requiring pickle only; drop unused scaler loading and dead `{{X_MIN}}` / `{{X_MAX}}` / `{{Y_MIN}}` / `{{Y_MAX}}` template replacements (those placeholders are not in the C++ template).

### Changed
- `docs/api/SCALERS_FORMAT.md`: rewritten for the raw I/O contract and Python-only scaler files; removed outdated migration / C++ JSON reader narrative.
- `README.md` and `docs/GETTING_STARTED.md`: link the end-to-end tutorial; clarify scaler file options.

### Added
- `docs/guides/END_TO_END_TUTORIAL.md`: pipeline from training through JAGS, with pointer to `demos/end-to-end-ddm.py`.
- `models/ddm.jnnx/metadata.json`: `description` field clarifying `transformations` vs ONNX runtime behavior.

## [1.0.0] - 2025-10-28

### Added
- Initial release of JNNX (JAGS Neural Network eXtension)
- Complete Python package structure with pip installation support
- Core classes: `JNNXPackage` and `JAGSModule`
- Command-line tools: `jnnx-setup`, `validate-jnnx`, `generate-module`, `validate-module`
- Comprehensive API documentation
- Installation guide and examples
- Support for ONNX model integration with JAGS
- Automatic C++ module generation from ONNX models
- Scaler support for input/output normalization
- Validation suite with 28 comprehensive tests
- Example models: SDT (Signal Detection Theory) and DDM (Drift Diffusion Model)
- Jupyter notebook workflows for model compilation and testing
- Integration with py2jags for Bayesian analysis
- Support for VectorFunction-based JAGS modules
- Global module instance for automatic registration
- Error handling and bounds checking
- MIT License

### Technical Details
- Python 3.8+ support
- JAGS 4.3.0+ compatibility
- ONNX Runtime 1.23.2+ integration
- C++17 compilation support
- Cross-platform Linux support
- Comprehensive test coverage
- Modern Python packaging with pyproject.toml

### Documentation
- Complete API documentation in `docs/api/API.md`
- Installation guide in `docs/guides/INSTALLATION.md`
- Examples and tutorials in `docs/examples/EXAMPLES.md`
- Project handoff memo with technical insights
- JAGS interface memo with common issues and solutions
- Contributing guidelines
- README with quick start guide

### Examples
- SDT model workflow notebook
- DDM model workflow notebook
- Complete integration examples
- Performance optimization examples
- Batch processing examples

## [Unreleased]

### Planned Features
- Support for additional neural network architectures
- Enhanced error reporting and debugging tools
- Performance optimizations for large models
- Additional scaler types beyond MinMaxScaler
- Integration with more Bayesian analysis tools
- Cross-platform support (Windows, macOS)
- Docker containerization support
- CI/CD pipeline improvements
