# Contributing to JNNX

Thank you for your interest in contributing to JNNX! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8+
- JAGS 4.3.0+
- ONNX Runtime 1.23.2+
- Git

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/jnnx.git
   cd jnnx
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   python tests/test-suite.py
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and small

### Testing

- Write tests for all new functionality
- Ensure all existing tests pass
- Aim for high test coverage
- Test both success and error cases

### Documentation

- Update relevant documentation when adding features
- Include examples in docstrings
- Update README.md if needed
- Add to API documentation in `docs/API.md`

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- Python version
- JAGS version
- ONNX Runtime version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

When requesting features, please include:

- Use case description
- Proposed API design
- Implementation considerations
- Examples of how it would be used

### Code Contributions

#### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   python tests/test-suite.py
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request** on GitHub

#### Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Request review from maintainers

### Documentation Contributions

- Fix typos and improve clarity
- Add examples and tutorials
- Improve installation instructions
- Update API documentation

## Project Structure

```
jnnx/
├── jnnx/                 # Main package
│   ├── __init__.py      # Package initialization
│   ├── core.py          # Core functionality
│   ├── utils.py         # Utility functions
│   ├── scripts/         # Command-line scripts
│   ├── templates/       # C++ templates
│   └── models/          # Example models
├── docs/                # Documentation
├── demos/               # Example notebooks
├── tests/               # Test suite
├── setup.py             # Package setup
├── pyproject.toml       # Modern Python packaging
└── README.md            # Project overview
```

## Release Process

Releases are managed by the maintainers. To request a release:

1. Ensure all tests pass
2. Update version numbers
3. Update CHANGELOG.md
4. Create a release tag
5. Build and upload to PyPI

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or inflammatory comments
- Personal attacks or political discussions
- Public or private harassment
- Publishing private information without permission
- Other unprofessional conduct

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers directly for sensitive issues

## License

By contributing to JNNX, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to JNNX!
