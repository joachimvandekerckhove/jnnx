"""
Core JNNX functionality for JAGS module generation and validation.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import onnxruntime as ort


class JNNXPackage:
    """Represents a .jnnx package containing ONNX model and metadata."""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.metadata = self._load_metadata()
        self.scalers = self._load_scalers()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata.json from package."""
        metadata_file = self.package_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.package_path}")
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _load_scalers(self) -> Dict[str, Any]:
        """Load scalers from package."""
        scalers_file = self.package_path / "scalers.pkl"
        if not scalers_file.exists():
            raise FileNotFoundError(f"scalers.pkl not found in {self.package_path}")
        
        with open(scalers_file, 'rb') as f:
            return pickle.load(f)
    
    @property
    def model_name(self) -> str:
        """Get the model name from metadata."""
        return self.metadata.get('model_name', 'unknown')
    
    @property
    def input_dim(self) -> int:
        """Get input dimension from metadata."""
        return len(self.metadata.get('input_parameters', []))
    
    @property
    def output_dim(self) -> int:
        """Get output dimension from metadata."""
        return len(self.metadata.get('output_parameters', []))
    
    def get_onnx_path(self) -> str:
        """Get absolute path to ONNX model file."""
        onnx_file = self.package_path / "model.onnx"
        if not onnx_file.exists():
            raise FileNotFoundError(f"model.onnx not found in {self.package_path}")
        return str(onnx_file.absolute())
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate package integrity."""
        errors = []
        
        # Check required files
        required_files = ['metadata.json', 'model.onnx', 'scalers.pkl']
        for file in required_files:
            if not (self.package_path / file).exists():
                errors.append(f"Missing required file: {file}")
        
        # Required metadata fields
        for field in [
            'model_name',
            'module_name',
            'function_name',
            'input_parameters',
            'output_parameters',
        ]:
            if field not in self.metadata:
                errors.append(f"Missing required metadata field: {field}")
        
        # Validate types for parameters
        if 'input_parameters' in self.metadata and not isinstance(self.metadata['input_parameters'], list):
            errors.append("'input_parameters' must be a list")
        if 'output_parameters' in self.metadata and not isinstance(self.metadata['output_parameters'], list):
            errors.append("'output_parameters' must be a list")
        
        # Check ONNX model
        try:
            session = ort.InferenceSession(self.get_onnx_path())
            input_shape = session.get_inputs()[0].shape
            output_shape = session.get_outputs()[0].shape
            
            if len(input_shape) != 2 or input_shape[0] != 1:
                errors.append("ONNX model must have input shape [1, N]")
            
            if len(output_shape) != 2 or output_shape[0] != 1:
                errors.append("ONNX model must have output shape [1, M]")
                
        except Exception as e:
            errors.append(f"ONNX model validation failed: {e}")
        
        return len(errors) == 0, errors
    
    def get_input_parameters(self) -> List[Dict[str, Any]]:
        """Get input parameter details."""
        return self.metadata.get('input_parameters', [])
    
    def get_output_parameters(self) -> List[Dict[str, Any]]:
        """Get output parameter details."""
        return self.metadata.get('output_parameters', [])
    
    def get_scaler_parameters(self) -> Dict[str, List[float]]:
        """Extract scaler parameters for C++ code generation."""
        scalers = self.scalers
        
        if 'x_min' in scalers:
            # Simple dictionary format
            return {
                'x_min': scalers['x_min'],
                'x_max': scalers['x_max'],
                'y_min': scalers['y_min'],
                'y_max': scalers['y_max']
            }
        else:
            # sklearn scaler format
            x_scaler = scalers.get('x_scaler')
            y_scaler = scalers.get('y_scaler')
            if x_scaler and y_scaler:
                return {
                    'x_min': x_scaler.data_min_.tolist(),
                    'x_max': x_scaler.data_max_.tolist(),
                    'y_min': y_scaler.data_min_.tolist(),
                    'y_max': y_scaler.data_max_.tolist()
                }
            else:
                raise ValueError("Invalid scaler format")
    
    def test_onnx_model(self, test_inputs: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Test the ONNX model with sample inputs."""
        session = ort.InferenceSession(self.get_onnx_path())
        
        if test_inputs is None:
            # Generate default test inputs
            input_params = self.get_input_parameters()
            test_inputs = []
            for _ in range(3):  # 3 test cases
                test_input = []
                for param in input_params:
                    min_val = param.get('min', 0)
                    max_val = param.get('max', 1)
                    test_input.append(np.random.uniform(min_val, max_val))
                test_inputs.append(np.array([test_input], dtype=np.float32))
        
        results = []
        for test_input in test_inputs:
            result = session.run(['output'], {'input': test_input})
            results.append(result[0])
        
        return {
            'model_path': self.get_onnx_path(),
            'input_shape': session.get_inputs()[0].shape,
            'output_shape': session.get_outputs()[0].shape,
            'test_results': results
        }


class JAGSModule:
    """Represents a generated JAGS module."""
    
    def __init__(self, package: JNNXPackage, build_dir: str):
        self.package = package
        self.build_dir = Path(build_dir)
        # Strict naming via metadata (required fields)
        # Required in metadata.json:
        #   module_name: the JAGS module name and install artifact basename
        #   function_name: the JAGS function exposed by the module
        metadata = self.package.metadata
        if 'module_name' not in metadata or not str(metadata['module_name']).strip():
            raise ValueError("metadata.json missing required field 'module_name'")
        if 'function_name' not in metadata or not str(metadata['function_name']).strip():
            raise ValueError("metadata.json missing required field 'function_name'")

        self.module_name = str(metadata['module_name']).strip()
        self.function_name = str(metadata['function_name']).strip()
        
    def generate_code(self) -> None:
        """Generate C++ module code and Makefile."""
        import shutil
        import subprocess
        
        # Create build directory
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ONNX model to build directory
        onnx_source = Path(self.package.get_onnx_path())
        onnx_dest = self.build_dir / "model.onnx"
        shutil.copy2(onnx_source, onnx_dest)
        
        # Copy ONNX external data file if it exists
        onnx_data_source = onnx_source.parent / "model.onnx.data"
        if onnx_data_source.exists():
            onnx_data_dest = self.build_dir / "model.onnx.data"
            shutil.copy2(onnx_data_source, onnx_data_dest)
            print(f"Copied external data file to {onnx_data_dest}")
        
        # Generate C++ source file
        cpp_file = self.build_dir / f"{self.module_name}.cc"
        self._generate_cpp_code(cpp_file)
        
        # Generate Makefile
        makefile = self.build_dir / "Makefile"
        self._generate_makefile(makefile)
    
    def _generate_cpp_code(self, output_file: Path) -> None:
        """Generate C++ module source code."""
        template_file = Path(__file__).parent / "templates" / "module.cc.template"
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        with open(template_file, 'r') as f:
            template = f.read()
        
        # Get template variables
        metadata = self.package.metadata
        input_params = metadata.get('input_parameters', [])
        output_params = metadata.get('output_parameters', [])
        
        # Replace template variables
        replacements = {
            '{{MODULE_NAME}}': self.module_name,
            '{{MODULE_CLASS}}': f"{self.module_name.replace('_', '').upper()}_Module",
            '{{FUNCTION_CLASS}}': f"{self.module_name.replace('_', '').upper()}_Function",
            '{{FUNCTION_NAME}}': self.function_name,
            '{{BANNER_STRING}}': f"JNNX Module {self.module_name} loaded successfully",
            '{{INPUT_DIM}}': str(len(input_params)),
            '{{OUTPUT_DIM}}': str(len(output_params)),
            '{{ONNX_PATH}}': str((self.build_dir / "model.onnx").absolute()),  # Use absolute path
        }
        
        # Add input/output limits
        if input_params:
            input_mins = [str(p.get('min', 0)) for p in input_params]
            input_maxs = [str(p.get('max', 1)) for p in input_params]
            replacements['{{INPUT_MIN}}'] = '{' + ', '.join(input_mins) + '}'
            replacements['{{INPUT_MAX}}'] = '{' + ', '.join(input_maxs) + '}'
        else:
            replacements['{{INPUT_MIN}}'] = '{0}'
            replacements['{{INPUT_MAX}}'] = '{1}'
        
        if output_params:
            output_mins = [str(p.get('min', 0)) for p in output_params]
            output_maxs = [str(p.get('max', 1)) for p in output_params]
            replacements['{{OUTPUT_MIN}}'] = '{' + ', '.join(output_mins) + '}'
            replacements['{{OUTPUT_MAX}}'] = '{' + ', '.join(output_maxs) + '}'
        else:
            replacements['{{OUTPUT_MIN}}'] = '{0}'
            replacements['{{OUTPUT_MAX}}'] = '{1}'
        
        # Apply replacements
        code = template
        for placeholder, value in replacements.items():
            code = code.replace(placeholder, value)
        
        with open(output_file, 'w') as f:
            f.write(code)
    
    def _generate_makefile(self, output_file: Path) -> None:
        """Generate Makefile for compilation."""
        template_file = Path(__file__).parent / "templates" / "Makefile.template"
        if not template_file.exists():
            raise FileNotFoundError(f"Makefile template not found: {template_file}")
        
        with open(template_file, 'r') as f:
            template = f.read()
        
        # Replace template variables
        # Choose a standard JAGS module installation directory for Linux
        install_dir = "/usr/lib/x86_64-linux-gnu/JAGS/modules-4/"

        replacements = {
            '{{MODULE_NAME}}': self.module_name,
            '{{SOURCE_FILE}}': f"{self.module_name}.cc",
            '{{INSTALL_DIR}}': install_dir,
        }
        
        # Apply replacements
        makefile_content = template
        for placeholder, value in replacements.items():
            makefile_content = makefile_content.replace(placeholder, value)
        
        # Normalize ONNX Runtime paths to absolute to avoid CWD issues
        project_root = Path(__file__).parent.parent
        abs_onnx_base = str((project_root / 'tmp' / 'onnxruntime-linux-x64-1.23.2').resolve())
        makefile_content = makefile_content.replace(
            '../../tmp/onnxruntime-linux-x64-1.23.2', abs_onnx_base
        )
        
        with open(output_file, 'w') as f:
            f.write(makefile_content)
    
    def compile(self) -> Tuple[bool, str]:
        """Compile the generated module.
        
        Returns:
            Tuple of (success, error_message)
        """
        makefile = self.build_dir / "Makefile"
        if not makefile.exists():
            return False, "Makefile not found. Run generate_code() first."
        
        import subprocess
        result = subprocess.run(['make'], cwd=self.build_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, ""
        else:
            error_msg = f"Compilation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            return False, error_msg
    
    def install(self) -> Tuple[bool, str]:
        """Install the compiled module.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check if .so file exists
        so_file = self.build_dir / f"{self.module_name}.so"
        if not so_file.exists():
            return False, f"Compiled module {so_file} not found. Run compile() first."
        
        import subprocess
        result = subprocess.run(['sudo', 'make', 'install'], cwd=self.build_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, ""
        else:
            error_msg = f"Installation failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            return False, error_msg


def validate_jnnx_package(package_path: str) -> Tuple[bool, List[str]]:
    """Validate a .jnnx package."""
    package = JNNXPackage(package_path)
    return package.validate()


def load_metadata(package_path: str) -> Dict[str, Any]:
    """Load metadata from a .jnnx package."""
    package = JNNXPackage(package_path)
    return package.metadata


def load_scalers(package_path: str) -> Dict[str, Any]:
    """Load scalers from a .jnnx package."""
    package = JNNXPackage(package_path)
    return package.scalers
