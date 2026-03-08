#!/usr/bin/env python3
"""
generate-module: Generate JAGS module code from templates and .jnnx metadata.

Usage: ./generate-module models/sdt.jnnx/
"""

import json
import sys
import pickle
from pathlib import Path


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


def extract_dimensions_from_metadata(metadata):
    """Extract input/output dimensions from metadata."""
    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])
    
    input_dim = len(input_params)
    output_dim = len(output_params)
    
    return input_dim, output_dim


def extract_limits_from_metadata(metadata):
    """Extract input/output limits from metadata."""
    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])
    
    input_min = [param.get('min', 0.0) for param in input_params]
    input_max = [param.get('max', 1.0) for param in input_params]
    output_min = [param.get('min', 0.0) for param in output_params]
    output_max = [param.get('max', 1.0) for param in output_params]
    
    return input_min, input_max, output_min, output_max


def format_array(arr):
    """Format array for C++ code generation."""
    if not arr:
        return "{}"
    
    formatted = []
    for val in arr:
        if isinstance(val, str):
            # Handle string representations of Inf/-Inf
            if val == "Inf" or val == "inf":
                formatted.append("1e38f")
            elif val == "-Inf" or val == "-inf":
                formatted.append("-1e38f")
            else:
                try:
                    float_val = float(val)
                    formatted.append(f"{float_val:.6f}f")
                except ValueError:
                    formatted.append(f"{val}f")
        else:
            if val == float('inf'):
                formatted.append("1e38f")
            elif val == float('-inf'):
                formatted.append("-1e38f")
            else:
                formatted.append(f"{val:.6f}f")
    
    return "{" + ", ".join(formatted) + "}"


def _templates_dir():
    """Directory containing C++/Makefile templates (works from repo or when installed)."""
    return Path(__file__).resolve().parent.parent / "templates"


def generate_module_code(metadata, scalers, onnx_file, output_dir):
    """Generate C++ module code from templates."""
    template_file = _templates_dir() / "module.cc.template"
    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        sys.exit(1)
    
    template_content = template_file.read_text()
    
    # Extract information from metadata
    # Require explicit names from metadata
    module_name = metadata.get('module_name')
    function_name = metadata.get('function_name')
    if not module_name or not function_name:
        print('Error: metadata.json must include module_name and function_name')
        sys.exit(1)
    function_class = f"{module_name.replace('_','').upper()}_Function"
    module_class = f"{module_name.replace('_','').upper()}_Module"
    
    input_dim, output_dim = extract_dimensions_from_metadata(metadata)
    input_min, input_max, output_min, output_max = extract_limits_from_metadata(metadata)
    
    # Extract scaler data
    x_min = scalers.get('x_min', [0.0] * input_dim)
    x_max = scalers.get('x_max', [1.0] * input_dim)
    y_min = scalers.get('y_min', [0.0] * output_dim)
    y_max = scalers.get('y_max', [1.0] * output_dim)
    
    # Create banner string
    model_name = metadata.get('model_name', module_name)
    banner = f"The {model_name} is being loaded. (c) 2025 Joachim Vandekerckhove"
    
    # Copy ONNX file to build directory for easier access
    onnx_copy = output_dir / "model.onnx"
    import shutil
    shutil.copy2(onnx_file, onnx_copy)
    print(f"Copied ONNX model to: {onnx_copy}")
    
    # Replace placeholders
    replacements = {
        '{{MODULE_NAME}}': module_name,
        '{{FUNCTION_NAME}}': function_name,
        '{{FUNCTION_CLASS}}': function_class,
        '{{MODULE_CLASS}}': module_class,
        '{{INPUT_DIM}}': str(input_dim),
        '{{OUTPUT_DIM}}': str(output_dim),
        '{{ONNX_PATH}}': str(onnx_copy.absolute()),
        '{{INPUT_MIN}}': format_array(input_min),
        '{{INPUT_MAX}}': format_array(input_max),
        '{{OUTPUT_MIN}}': format_array(output_min),
        '{{OUTPUT_MAX}}': format_array(output_max),
        '{{BANNER_STRING}}': banner,
        '{{X_MIN}}': format_array(x_min),
        '{{X_MAX}}': format_array(x_max),
        '{{Y_MIN}}': format_array(y_min),
        '{{Y_MAX}}': format_array(y_max)
    }
    
    # Apply replacements
    generated_content = template_content
    for placeholder, value in replacements.items():
        generated_content = generated_content.replace(placeholder, value)
    
    # Write generated file
    output_file = output_dir / f"{module_name}.cc"
    output_file.write_text(generated_content)
    
    print(f"Generated: {output_file}")
    
    return output_file


def generate_makefile(metadata, output_dir):
    """Generate Makefile from template."""
    template_file = _templates_dir() / "Makefile.template"
    if not template_file.exists():
        print(f"Error: Makefile template not found: {template_file}")
        sys.exit(1)
    
    template_content = template_file.read_text()
    
    # Extract information from metadata
    module_name = metadata.get('module_name')
    if not module_name:
        print('Error: metadata.json must include module_name')
        sys.exit(1)
    
    # Default installation directory
    install_dir = "/usr/lib/x86_64-linux-gnu/JAGS/modules-4/"
    
    # Replace placeholders
    replacements = {
        '{{MODULE_NAME}}': module_name,
        '{{INSTALL_DIR}}': install_dir
    }
    
    # Apply replacements
    generated_content = template_content
    for placeholder, value in replacements.items():
        generated_content = generated_content.replace(placeholder, value)
    
    # Write generated file
    output_file = output_dir / "Makefile"
    output_file.write_text(generated_content)
    
    print(f"Generated: {output_file}")
    
    return output_file


def ensure_onnxruntime_in_tmp():
    """Ensure ONNX Runtime is available in tmp directory."""
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)
    
    onnx_dir = tmp_dir / 'onnxruntime-linux-x64-1.23.2'
    if not onnx_dir.exists():
        print("ONNX Runtime not found in tmp/, extracting...")
        tgz_file = tmp_dir / 'onnxruntime-linux-x64-1.23.2.tgz'
        if tgz_file.exists():
            import tarfile
            with tarfile.open(tgz_file, 'r:gz') as tar:
                tar.extractall(tmp_dir)
            print(f"✓ ONNX Runtime extracted to {onnx_dir}")
        else:
            print("⚠ Warning: ONNX Runtime archive not found in tmp/")
            print("  You may need to download it manually")
    else:
        print(f"✓ ONNX Runtime found in {onnx_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: ./generate-module <jnnx-directory>")
        print("Example: ./generate-module models/sdt.jnnx/")
        sys.exit(1)
    
    jnnx_dir = sys.argv[1]
    
    # Find required files
    metadata_file, onnx_file, scalers_file = find_files(jnnx_dir)
    print(f"Found files:")
    print(f"  Metadata: {metadata_file}")
    print(f"  ONNX: {onnx_file}")
    print(f"  Scalers: {scalers_file}")
    print()
    
    # Load metadata and scalers
    metadata = load_metadata(metadata_file)
    scalers = load_scalers(scalers_file)
    print(f"Metadata loaded: {metadata.get('model_name', 'unnamed')}")
    print(f"Scalers loaded: {len(scalers)} parameters")
    print()
    
    # Ensure ONNX Runtime is available in tmp/
    ensure_onnxruntime_in_tmp()
    print()
    
    # Create output directory in tmp/ (per constitution.md)
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)
    output_dir = tmp_dir / f"{Path(jnnx_dir).name}_build"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate module code
    print("Generating module code...")
    module_file = generate_module_code(metadata, scalers, onnx_file, output_dir)
    print()
    
    # Generate Makefile
    print("Generating Makefile...")
    makefile = generate_makefile(metadata, output_dir)
    print()
    
    print("=" * 60)
    print("Module generation complete!")
    print()
    print("To compile and install:")
    print(f"  cd {output_dir}")
    print("  make")
    print("  sudo make install")


if __name__ == "__main__":
    main()