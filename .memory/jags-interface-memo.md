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

### 2. **Variable Addressing Issues**

#### Problem: Parameter names in JAGS output
- **Issue**: Expected `output[1]`, `output[2]`, `output[3]` but got `output_1`, `output_2`, `output_3`
- **Solution**: Use underscore notation for array elements in parameter names
- **Pattern**: `f'output_{i+1}'` instead of `f'output[{i+1}]'`

### 3. **JAGS Model Compilation Issues**

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
    output[1:3] <- ddm(boundary, drift, ndt)
    # Use output_1, output_2, output_3 for monitoring
}
```

#### Parameter Access
```python
# Correct way to access array parameters
for i in range(3):
    param_name = f'output_{i+1}'
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

### 11. **Remember**

- JAGS is picky about syntax and data types
- `py2jags` handles file management - don't interfere
- Always test with simple cases first
- Parameter names in JAGS use underscores, not brackets
- Module names must be consistent throughout the pipeline
- Scaling parameters must match model dimensions exactly
- Empty data files cause silent failures
