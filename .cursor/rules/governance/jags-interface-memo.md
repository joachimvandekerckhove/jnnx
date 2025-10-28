# JAGS Interface Memo
## Common issues when trying to interact with JAGS

### 1. **Script Calling Issues**

#### Problem: "calling jags wrong -- you need to make files for the model and the data, you can't put the model in the script file"
- **Issue**: Initially tried to pass model code directly to JAGS
- **Solution**: Use `py2jags` which automatically creates separate model and data files
- **Key**: `py2jags` handles file creation behind the scenes - don't try to manage JAGS files manually

#### Problem: "jags fails because the data.S is empty"
- **Issue**: JAGS fails silently when data file is empty
- **Solution**: Always ensure data dictionary has at least one non-empty value
- **Example**: Add `'n': 1` to prevent empty `data.S` file

### 2. **py2jags API Details**

#### Correct py2jags API Usage
```python
import py2jags

# CORRECT: Use run_jags() function
chains = py2jags.run_jags(
    model_string=model_code, 
    data_dict={'n': 1}, 
    nchains=1, 
    nsamples=1, 
    nadapt=0, 
    nburnin=0
)

# INCORRECT: These don't work as expected
# chains = py2jags.Model(...)  # Wrong API
# chains = py2jags.JagsRunner(...)  # Wrong API  
# chains = py2jags.McmcSamples(...)  # Wrong API
```

#### py2jags.run_jags() Parameters
- **model_string**: JAGS model code as string
- **data_dict**: Dictionary with data (must be non-empty)
- **nchains**: Number of chains (default: 1)
- **nsamples**: Number of samples (default: 1000)
- **nadapt**: Adaptation samples (default: 1000)
- **nburnin**: Burn-in samples (default: 1000)
- **modules**: List of module names to load (e.g., `['sdt_emulator']`)

#### Parameter Access Pattern
```python
# Access array parameters using underscore notation
for i in range(output_dim):
    param_name = f'result_{i+1}'  # result_1, result_2, result_3, etc.
    if param_name in chains.parameter_names:
        value = chains.get_samples(param_name)[0]
```

#### Complete Validation Pattern
```python
# Standard pattern for testing JAGS modules
model_code = f'''
model {{
    result <- {function_name}({arg1}, {arg2}, {arg3}, {arg4})
    dummy ~ dnorm(0, 1)
}}
'''

data = {'n': 1}  # Non-empty data dictionary

chains = py2jags.run_jags(
    model_string=model_code, 
    data_dict=data, 
    nchains=1, 
    nsamples=1, 
    nadapt=0, 
    nburnin=0,
    monitorparams=['result'],
    modules=['{module_name}']  # Load custom module
)

# Extract results
results = []
for i in range(output_dim):
    param_name = f'result_{i+1}'
    if param_name in chains.parameter_names:
        results.append(chains.get_samples(param_name)[0])
```

### 3. **Variable Addressing Issues**

#### Problem: Parameter names in JAGS output
- **Issue**: Expected `output[1]`, `output[2]`, `output[3]` but got `output_1`, `output_2`, `output_3`
- **Solution**: Use underscore notation for array elements in parameter names
- **Pattern**: `f'output_{i+1}'` instead of `f'output[{i+1}]'`

### 4. **JAGS Model Compilation Issues**

#### Problem: "Deleting model" - JAGS fails to compile
- **Issue**: Model syntax errors or incompatible data types
- **Solution**: 
  - Use simple test models first
  - Ensure data types match model expectations
  - Check for undefined variables in model

### 5. **Module Loading Issues**

#### Problem: Module loading hangs
- **Issue**: JAGS doesn't flush output after loading
- **Solution**: Pipe 'exit' after 'load <module>' command

#### Problem: Loading custom modules in py2jags
- **Issue**: Need to load custom JAGS modules (like sdt_emulator) in py2jags
- **Solution**: Use the `modules` parameter in `py2jags.run_jags()`
- **Example**:
```python
chains = py2jags.run_jags(
    model_string=model_code,
    data_dict={'n': 1},
    nchains=1,
    nsamples=1,
    modules=['sdt_emulator']  # Load custom module
)
```
- **Key**: Module names should match the module name (not function name)
- **Verification**: Check JAGS output for "Loading module: sdt_emulator: ok"

### 6. **Function Signature Issues**

#### Problem: "Too many arguments" error
- **Issue**: Function signature mismatch between C++ module and JAGS call
- **Solution**: 
  - Ensure C++ template uses correct input/output dimensions
  - Verify function registration matches expected signature
  - Check that scaling parameters match model dimensions

### 7. **Data Structure Issues**

#### Problem: Empty or malformed data dictionaries
- **Issue**: Missing required data fields or incorrect data types
- **Solution**: 
  - Always include at least one data field
  - Use correct data types (scalars vs arrays)
  - Ensure data matches model expectations

### 8. **Best Practices for JAGS Interface**

#### Debugging Tips
- Check `chains.parameter_names` to see available parameters
- Use `verbosity=0` to reduce noise, increase for debugging
- Check JAGS temp directories for error logs
- Use simple models to isolate issues

### 9. **Common Patterns**

#### JAGS Model Code
```jags
model {
    result <- ddm4(boundary, drift, ndt, bias)
    dummy ~ dnorm(0, 1)  # Required for CODA output generation
    # Use result_1, result_2, result_3, result_4, result_5 for monitoring
}
```

#### Parameter Access
```python
# Correct way to access array parameters
for i in range(5):  # For 5D output
    param_name = f'result_{i+1}'
    if param_name in chains.parameter_names:
        value = chains.get_samples(param_name)[0]
```

### 10. **Troubleshooting Checklist**

1. ✅ Is the data dictionary non-empty?
2. ✅ Are module names consistent between registration and loading?
3. ✅ Do scaling parameters match model dimensions?
4. ✅ Is the function name using correct case and format?
5. ✅ Are parameter names using underscore notation for arrays?
6. ✅ Are input/output dimensions correct in C++ template?
7. ✅ Is the JAGS model syntax valid?
8. ✅ Are data types compatible with model expectations?
9. ✅ Is the module compiled and installed correctly?
10. ✅ Are you using `py2jags.run_jags()` instead of other py2jags classes?
11. ✅ Does your model include a stochastic node (like `dummy ~ dnorm(0, 1)`) for CODA output?
12. ✅ Are you using the `modules` parameter to load custom modules?
13. ✅ Does JAGS output show "Loading module: [module_name]: ok"?

### 11. **Critical Discovery: ONNX Model Scaling**

**IMPORTANT**: ONNX models that were trained with scaling have the scaling **built into the model itself**. The JAGS module should **NOT** apply additional scaling.

#### Correct Approach:
- Pass inputs directly to ONNX (no scaling)
- Use ONNX outputs directly (no denormalization)
- The scaling parameters in `.jnnx` packages are for training, not inference

#### Wrong Approach (causes double-scaling):
- Apply input scaling before ONNX
- Apply output denormalization after ONNX
- This results in incorrect values

#### Verification:
```python
# Test that JAGS and Python ONNX give identical results
import py2jags
import onnxruntime as ort
import numpy as np

# Both should give identical results
jags_result = py2jags.run_jags(..., monitorparams=['result'])
onnx_result = ort.InferenceSession('model.onnx').run(['output'], {'input': input_data})
```

### 12. **Remember**

- JAGS is picky about syntax and data types
- `py2jags` handles file management - don't interfere
- Always test with simple cases first
- Parameter names in JAGS use underscores, not brackets
- Module names must be consistent throughout the pipeline
- Scaling parameters must match model dimensions exactly
- Empty data files cause silent failures
- **Use `py2jags.run_jags()` - other py2jags classes don't work as expected**
- **Always include a stochastic node in your model for CODA output generation**
- **Check `chains.parameter_names` to verify available parameters**
- **Use the `modules` parameter to load custom JAGS modules**
- **Module names in `modules` parameter should match the module name, not function name**
- **ONNX models handle scaling internally - don't apply additional scaling in JAGS modules**
