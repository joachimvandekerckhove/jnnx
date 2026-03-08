#!/usr/bin/env python3
"""
validate-module: Test suite to validate a compiled JAGS module.

Usage: ./validate-module models/sdt.jnnx/

Tests:
1) Neural network loads in JAGS without error
2) Calling with valid input vector works without error for a range of values
3) Calling with valid input vector returns valid output vector
4) Calling with invalid input vector (wrong size) triggers error
5) Calling with invalid input vector (out of bounds) triggers error
6) Output vector is numerically identical to direct Python evaluation
"""

import json
import sys
import numpy as np
import pickle
from pathlib import Path

try:
    import py2jags
    import onnxruntime as ort
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install py2jags onnxruntime")
    sys.exit(1)


def find_files(jnnx_dir):
    """Find required files in .jnnx directory."""
    jnnx_path = Path(jnnx_dir)
    if not jnnx_path.exists():
        print(f"Error: Directory {jnnx_dir} does not exist")
        sys.exit(1)
    
    if not jnnx_path.name.endswith('.jnnx'):
        print(f"Error: Directory {jnnx_dir} does not end with .jnnx")
        sys.exit(1)
    
    # Find metadata.json file
    metadata_file = jnnx_path / "metadata.json"
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {jnnx_dir}")
        sys.exit(1)
    
    # Find model.onnx file
    onnx_file = jnnx_path / "model.onnx"
    if not onnx_file.exists():
        print(f"Error: model.onnx not found in {jnnx_dir}")
        sys.exit(1)
    
    # Find scalers.pkl file
    scalers_file = jnnx_path / "scalers.pkl"
    if not scalers_file.exists():
        print(f"Error: scalers.pkl not found in {jnnx_dir}")
        sys.exit(1)
    
    return metadata_file, onnx_file, scalers_file


def load_metadata(metadata_file):
    """Load metadata.json configuration."""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {metadata_file}: {e}")
        sys.exit(1)


def load_scalers(scalers_file):
    """Load scalers from pickle file."""
    try:
        with open(scalers_file, 'rb') as f:
            scalers_data = pickle.load(f)
        
        # Handle sklearn MinMaxScaler format
        if isinstance(scalers_data, dict) and 'x_scaler' in scalers_data and 'y_scaler' in scalers_data:
            x_scaler = scalers_data['x_scaler']
            y_scaler = scalers_data['y_scaler']
            return {
                'x_min': x_scaler.data_min_.tolist(),
                'x_max': x_scaler.data_max_.tolist(),
                'y_min': y_scaler.data_min_.tolist(),
                'y_max': y_scaler.data_max_.tolist()
            }
        # Handle simple dictionary format
        elif isinstance(scalers_data, dict) and 'x_min' in scalers_data:
            return scalers_data
        else:
            raise ValueError("Unknown scaler format")
            
    except Exception as e:
        print(f"Error: Could not load scalers from {scalers_file}: {e}")
        sys.exit(1)


def test_module_loading(jnnx_dir):
    """Test 1: Module loads in JAGS without error."""
    print("Test 1: Module loading...")
    
    try:
        # Extract function name from metadata
        metadata_file = Path(jnnx_dir) / "metadata.json"
        metadata = load_metadata(metadata_file)
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        # Create a simple JAGS model to test loading
        model_code = f'''
        model {{
            result <- {function_name}(1.0, 1.0)
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
            modules=[module_name]  # Load the custom module
        )
        
        print(f"  ✓ Module loaded successfully in JAGS")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading module: {e}")
        return False


def test_valid_input_range(jnnx_dir):
    """Test 2: Valid input range works without error."""
    print("Test 2: Valid input range...")
    
    try:
        # Load metadata to get function name and dimensions
        metadata_file = Path(jnnx_dir) / "metadata.json"
        metadata = load_metadata(metadata_file)
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        input_params = metadata.get('input_parameters', [])
        input_dim = len(input_params)
        
        # Create test inputs within valid ranges
        test_inputs = []
        for param in input_params:
            min_val = param.get('min', 0.0)
            max_val = param.get('max', 1.0)
            mid_val = (min_val + max_val) / 2.0
            test_inputs.append(mid_val)
        
        # Create JAGS model with test inputs
        input_str = ', '.join([str(x) for x in test_inputs])
        model_code = f'''
        model {{
            result <- {function_name}({input_str})
            dummy ~ dnorm(0, 1)
        }}
        '''
        
        data = {'n': 1}
        
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=data,
            nchains=1,
            nsamples=1,
            nadapt=0,
            nburnin=0,
            modules=[module_name]  # Load the custom module
        )
        
        print(f"  ✓ Valid input range works: {test_inputs}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error with valid input range: {e}")
        return False


def test_valid_output(jnnx_dir):
    """Test 3: Valid output vector."""
    print("Test 3: Valid output...")
    
    try:
        # Load metadata
        metadata_file = Path(jnnx_dir) / "metadata.json"
        metadata = load_metadata(metadata_file)
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        input_params = metadata.get('input_parameters', [])
        output_params = metadata.get('output_parameters', [])
        input_dim = len(input_params)
        output_dim = len(output_params)
        
        # Create test inputs
        test_inputs = []
        for param in input_params:
            min_val = param.get('min', 0.0)
            max_val = param.get('max', 1.0)
            mid_val = (min_val + max_val) / 2.0
            test_inputs.append(mid_val)
        
        # Create JAGS model
        input_str = ', '.join([str(x) for x in test_inputs])
        model_code = f'''
        model {{
            result <- {function_name}({input_str})
            dummy ~ dnorm(0, 1)
        }}
        '''
        
        data = {'n': 1}
        
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=data,
            nchains=1,
            nsamples=1,
            nadapt=0,
            nburnin=0,
            modules=[module_name]  # Load the custom module
        )
        
        # Extract results
        results = []
        for i in range(output_dim):
            param_name = f'result_{i+1}'
            if param_name in chains.parameter_names:
                results.append(chains.get_samples(param_name)[0])
        
        # Check output validity
        if len(results) != output_dim:
            print(f"  ✗ Output dimension mismatch: expected {output_dim}, got {len(results)}")
            return False
        
        if np.any(np.isnan(results)) or np.any(np.isinf(results)):
            print(f"  ✗ Invalid output values (NaN or Inf): {results}")
            return False
        
        print(f"  ✓ Valid output: {results}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error with valid output: {e}")
        return False


def test_invalid_input_size(jnnx_dir):
    """Test 4: Invalid input size triggers error."""
    print("Test 4: Invalid input size...")
    
    try:
        # Load metadata
        metadata_file = Path(jnnx_dir) / "metadata.json"
        metadata = load_metadata(metadata_file)
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        # Create JAGS model with wrong number of arguments
        model_code = f'''
        model {{
            result <- {function_name}(1.0, 2.0, 3.0, 4.0, 5.0)
            dummy ~ dnorm(0, 1)
        }}
        '''
        
        data = {'n': 1}
        
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=data,
            nchains=1,
            nsamples=1,
            nadapt=0,
            nburnin=0,
            modules=[module_name]  # Load the custom module
        )
        
        print(f"  ✗ Invalid input size did not trigger error (unexpected)")
        return False
        
    except Exception as e:
        print(f"  ✓ Invalid input size correctly triggered error: {e}")
        return True


def test_invalid_input_bounds(jnnx_dir):
    """Test 5: Invalid input bounds triggers error."""
    print("Test 5: Invalid input bounds...")
    
    try:
        # Load metadata
        metadata_file = Path(jnnx_dir) / "metadata.json"
        metadata = load_metadata(metadata_file)
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        input_params = metadata.get('input_parameters', [])
        
        # Create out-of-bounds inputs
        test_inputs = []
        for param in input_params:
            min_val = param.get('min', 0.0)
            max_val = param.get('max', 1.0)
            # Use value way outside bounds
            out_of_bounds = max_val + 100.0
            test_inputs.append(out_of_bounds)
        
        # Create JAGS model
        input_str = ', '.join([str(x) for x in test_inputs])
        model_code = f'''
        model {{
            result <- {function_name}({input_str})
            dummy ~ dnorm(0, 1)
        }}
        '''
        
        data = {'n': 1}
        
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=data,
            nchains=1,
            nsamples=1,
            nadapt=0,
            nburnin=0,
            modules=[module_name]  # Load the custom module
        )
        
        print(f"  ✗ Invalid input bounds did not trigger error (unexpected)")
        return False
        
    except Exception as e:
        print(f"  ✓ Invalid input bounds correctly triggered error: {e}")
        return True


def test_numerical_consistency(jnnx_dir):
    """Test 6: Numerical consistency with Python evaluation."""
    print("Test 6: Numerical consistency...")
    
    try:
        # Load metadata and scalers
        metadata_file = Path(jnnx_dir) / "metadata.json"
        onnx_file = Path(jnnx_dir) / "model.onnx"
        scalers_file = Path(jnnx_dir) / "scalers.pkl"
        
        metadata = load_metadata(metadata_file)
        scalers = load_scalers(scalers_file)
        
        module_name = metadata.get('module_name')
        function_name = metadata.get('function_name')
        if not module_name or not function_name:
            raise ValueError('metadata.json must include module_name and function_name')
        
        input_params = metadata.get('input_parameters', [])
        output_params = metadata.get('output_parameters', [])
        input_dim = len(input_params)
        output_dim = len(output_params)
        
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_file))
        
        # Create test inputs
        test_inputs = []
        for param in input_params:
            min_val = param.get('min', 0.0)
            max_val = param.get('max', 1.0)
            mid_val = (min_val + max_val) / 2.0
            test_inputs.append(mid_val)
        
        # Python evaluation with identical scaling
        x_min = np.array(scalers['x_min'])
        x_max = np.array(scalers['x_max'])
        y_min = np.array(scalers['y_min'])
        y_max = np.array(scalers['y_max'])
        
        input_scaled = (np.array(test_inputs) - x_min) / (x_max - x_min)
        input_tensor = input_scaled.astype(np.float32).reshape(1, -1)
        outputs = session.run(None, {'input': input_tensor})
        python_output = outputs[0][0] * (y_max - y_min) + y_min
        
        # JAGS evaluation
        input_str = ', '.join([str(x) for x in test_inputs])
        model_code = f'''
        model {{
            result <- {function_name}({input_str})
            dummy ~ dnorm(0, 1)
        }}
        '''
        
        data = {'n': 1}
        
        chains = py2jags.run_jags(
            model_string=model_code,
            data_dict=data,
            nchains=1,
            nsamples=1,
            nadapt=0,
            nburnin=0,
            modules=[module_name]  # Load the custom module
        )
        
        # Extract JAGS results
        jags_output = []
        for i in range(output_dim):
            param_name = f'result_{i+1}'
            if param_name in chains.parameter_names:
                jags_output.append(chains.get_samples(param_name)[0])
        
        # Compare results
        max_diff = 0.0
        for i in range(output_dim):
            diff = abs(python_output[i] - jags_output[i])
            max_diff = max(max_diff, diff)
        
        tolerance = 1e-6
        if max_diff < tolerance:
            print(f"  ✓ Numerical consistency: max difference = {max_diff:.2e}")
            return True
        else:
            print(f"  ✗ Numerical inconsistency: max difference = {max_diff:.2e}")
            print(f"    Python: {python_output}")
            print(f"    JAGS:   {jags_output}")
            return False
        
    except Exception as e:
        print(f"  ✗ Error in numerical consistency test: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: ./validate-module <jnnx-directory>")
        print("Example: ./validate-module models/sdt.jnnx/")
        sys.exit(1)
    
    jnnx_dir = sys.argv[1]
    
    # Find required files
    metadata_file, onnx_file, scalers_file = find_files(jnnx_dir)
    print(f"Validating module for: {jnnx_dir}")
    print(f"  Metadata: {metadata_file}")
    print(f"  ONNX: {onnx_file}")
    print(f"  Scalers: {scalers_file}")
    print()
    
    # Load configuration
    metadata = load_metadata(metadata_file)
    print(f"Model: {metadata.get('model_name', 'unnamed')}")
    print(f"Module: {metadata.get('module_name', 'unset')}  Function: {metadata.get('function_name', 'unset')}")
    print(f"Version: {metadata.get('version', 'unknown')}")
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 6
    
    if test_module_loading(jnnx_dir):
        tests_passed += 1
    
    if test_valid_input_range(jnnx_dir):
        tests_passed += 1
    
    if test_valid_output(jnnx_dir):
        tests_passed += 1
    
    if test_invalid_input_size(jnnx_dir):
        tests_passed += 1
    
    if test_invalid_input_bounds(jnnx_dir):
        tests_passed += 1
    
    if test_numerical_consistency(jnnx_dir):
        tests_passed += 1
    
    # Summary
    print()
    print("=" * 50)
    print(f"Validation Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Module is working correctly.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please check the module.")
        sys.exit(1)


if __name__ == "__main__":
    main()