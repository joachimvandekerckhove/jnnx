# JNNX Project Handoff Memo

## Project Overview
**Goal**: Create a system to evaluate trained neural networks (ONNX/PTH files) as deterministic nodes within JAGS Bayesian models.

**Status**: All phases implemented and tested successfully.

## Critical Technical Insights

### 1. ONNX Model Scaling Architecture
**MOST IMPORTANT DISCOVERY**: ONNX models trained with scaling have scaling built into the model itself.

- CORRECT: Pass inputs directly to ONNX, use outputs directly
- WRONG: Apply additional input scaling or output denormalization

**Verification Pattern**:
```python
import onnxruntime as ort
import numpy as np

# Test raw ONNX behavior first
session = ort.InferenceSession('model.onnx')
test_input = np.array([[0.0, 0.0]], dtype=np.float32)
result = session.run(['output'], {'input': test_input})
# Use these values as ground truth for JAGS module
```

### 2. py2jags API Requirements
**Essential Parameters**:
- `monitorparams=['result']` (NOT `monitor=['result']`)
- `modules=['module_name']` for custom JAGS modules
- Always include stochastic node: `dummy ~ dnorm(0, 1)`

**Complete Pattern**:
```python
result = py2jags.run_jags(
    model_string=model_code,
    data_dict={'n': 1},  # Non-empty data required
    nchains=1, nsamples=1, nadapt=0, nburnin=0,
    monitorparams=['result'],  # Monitor the function output
    modules=['module_name']    # Load custom module
)
```

### 3. JAGS Module Architecture
**Modern Pattern**:
- Use `VectorFunction` (not `ScalarFunction`) for multi-output models
- Global module instance for automatic registration
- No `extern "C"` or `load_module()` needed

**Template Structure**:
```cpp
class ModelName_Function : public VectorFunction {
    // ONNX Runtime session
    // evaluate() method with direct ONNX calls
};

class ModelName_Module : public Module {
    // Constructor inserts function
};

// Global instance for auto-registration
ModelName_Module _modelname_module;
```

## Implementation Strategy

### Phase 0: Foundation (1-2 days)
1. Test ONNX model behavior with raw inputs/outputs
2. Master py2jags API - run examples, understand CODA requirements
3. Study existing JAGS modules for patterns
4. Analyze target models - dimensions, scaling, bounds

### Phase 1: Core Implementation (2-3 days)
1. Start with VectorFunction - skip ScalarFunction entirely
2. No scaling in JAGS module - pass inputs directly to ONNX
3. Use real ONNX files in tests - no dummy content
4. Simple validation - one model, basic functionality

### Phase 2: Robust System (3-4 days)
1. Template system - flexible, versioned templates
2. Complete test suite - real models, edge cases
3. Error handling - JAGS-style bounds checking
4. Documentation - examples, troubleshooting

### Phase 3: Production (2-3 days)
1. Multiple model support - different dimensions
2. Installation automation - scripts, dependencies
3. Performance optimization - memory, error handling
4. Comprehensive validation - integration tests

## Key Files and Structure

### Core Scripts
- `scripts/jnnx-setup.py` - JSON configuration interface
- `scripts/validate-jnnx.py` - Package validation
- `scripts/generate-module.py` - C++ code generation
- `scripts/validate-module.py` - JAGS integration testing

### Templates
- `jnnx/templates/module.cc.template` - C++ module template
- `jnnx/templates/Makefile.template` - Build configuration

### Configuration
- `.jnnx` format specification in `docs/governance/jnnx-format-spec.md`
- JAGS interface memo in `.cursor/rules/governance/jags-interface-memo.md`

## Common Pitfalls to Avoid

### 1. Double-Scaling Trap
- Don't apply scaling in JAGS module if ONNX model already handles it
- Test raw ONNX behavior first, use as ground truth

### 2. py2jags Parameter Confusion
- `monitor=['result']` - causes "no monitors" error
- `monitorparams=['result']` - correct parameter

### 3. Missing Stochastic Nodes
- Models without stochastic nodes fail CODA generation
- Always include `dummy ~ dnorm(0, 1)`

### 4. Test Implementation Details
- Testing for `extern "C"` or `load_module()`
- Testing for actual functionality and behavior

### 5. Dummy Test Data
- Using dummy ONNX files in tests
- Using real ONNX models for validation

## Validation Success Criteria

### Numerical Consistency
JAGS module outputs must match Python ONNX outputs exactly:
```python
# Both should give identical results (within machine precision)
jags_result = py2jags.run_jags(..., monitorparams=['result'])
onnx_result = ort.InferenceSession('model.onnx').run(['output'], {'input': input_data})
# Differences should be < 1e-6
```

### Functional Requirements
- Module loads: "Loading module: [name]: ok"
- Function recognized: `function_name()` accessible in JAGS
- Model compiles and executes
- Error handling for invalid inputs
- Bounds checking works correctly

## Dependencies and Setup

### Required Software
- JAGS 4.3.0+
- ONNX Runtime 1.23.2+ (Linux x64)
- py2jags 0.1.0+
- Python 3.11+ with scikit-learn

### Installation Paths
- ONNX Runtime: `tmp/onnxruntime-linux-x64-1.23.2/`
- JAGS modules: `/usr/lib/x86_64-linux-gnu/JAGS/modules-4/`

## Success Metrics

The project is complete when:
1. All 28 tests pass in `tests/test-suite.py`
2. Perfect numerical consistency between JAGS and Python ONNX
3. All 5 phases implemented and validated
4. Real-world models work (SDT, DDM4 examples)

## Final Notes

**The most critical insight**: ONNX models handle scaling internally. This single discovery eliminates weeks of debugging and leads to clean, correct implementations.

**Start with understanding, not implementation**. Test ONNX behavior first, master py2jags API, then build the JAGS module. The implementation will be straightforward once the architecture is understood.

**This project is complete and production-ready.** All phases work correctly, tests pass, and the system successfully evaluates neural networks within JAGS Bayesian models.
