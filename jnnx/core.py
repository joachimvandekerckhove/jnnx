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


class JAGSModule:
    """Represents a generated JAGS module."""
    
    def __init__(self, package: JNNXPackage, build_dir: str):
        self.package = package
        self.build_dir = Path(build_dir)
        self.module_name = f"{package.model_name}_emulator"
        
    def generate_code(self) -> None:
        """Generate C++ module code and Makefile."""
        # This would contain the logic from generate-module script
        # For now, we'll assume it's handled by the command-line tool
        pass
    
    def compile(self) -> bool:
        """Compile the generated module."""
        makefile = self.build_dir / "Makefile"
        if not makefile.exists():
            raise FileNotFoundError("Makefile not found. Run generate_code() first.")
        
        import subprocess
        result = subprocess.run(['make'], cwd=self.build_dir, capture_output=True, text=True)
        return result.returncode == 0
    
    def install(self) -> bool:
        """Install the compiled module."""
        import subprocess
        result = subprocess.run(['sudo', 'make', 'install'], cwd=self.build_dir, capture_output=True, text=True)
        return result.returncode == 0


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
