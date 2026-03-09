"""
Utility functions for JNNX package.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import onnxruntime as ort


def find_jnnx_packages(directory: str) -> List[str]:
    """Find all .jnnx packages in a directory."""
    dir_path = Path(directory)
    packages = []
    
    for item in dir_path.iterdir():
        if item.is_dir() and item.name.endswith('.jnnx'):
            packages.append(str(item))
    
    return packages


def get_package_info(package_path: str) -> Dict[str, Any]:
    """Get basic information about a .jnnx package."""
    package_path = Path(package_path)
    
    info = {
        'name': package_path.name,
        'path': str(package_path),
        'files': [],
        'size': 0
    }
    
    for file in package_path.rglob('*'):
        if file.is_file():
            info['files'].append(str(file.relative_to(package_path)))
            info['size'] += file.stat().st_size
    
    return info


def create_jnnx_package(output_path: str, model_name: str, 
                       onnx_path: str, scalers: Dict[str, Any],
                       input_params: List[Dict], output_params: List[Dict]) -> None:
    """Create a new .jnnx package."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.json
    metadata = {
        'model_name': model_name,
        'version': '1.0.0',
        'input_parameters': input_params,
        'output_parameters': output_params,
        'transformations': {
            'input_scaling': 'MinMaxScaler',
            'output_scaling': 'MinMaxScaler'
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy ONNX model
    import shutil
    shutil.copy2(onnx_path, output_path / 'model.onnx')
    
    # Save scalers
    with open(output_path / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    # Create scalers.txt
    scalers_txt = output_path / 'scalers.txt'
    with open(scalers_txt, 'w') as f:
        if 'x_min' in scalers:
            # Simple dictionary format
            for key in ['x_min', 'x_max', 'y_min', 'y_max']:
                for val in scalers[key]:
                    f.write(f"{val}\n")
        else:
            # sklearn scaler format
            x_scaler = scalers.get('x_scaler')
            y_scaler = scalers.get('y_scaler')
            if x_scaler and y_scaler:
                for val in x_scaler.data_min_:
                    f.write(f"{val}\n")
                for val in x_scaler.data_max_:
                    f.write(f"{val}\n")
                for val in y_scaler.data_min_:
                    f.write(f"{val}\n")
                for val in y_scaler.data_max_:
                    f.write(f"{val}\n")


def test_onnx_model(onnx_path: str, test_inputs: List[np.ndarray]) -> Dict[str, Any]:
    """Test an ONNX model with given inputs. Inputs should be in raw (original) domain per the scaling contract."""
    session = ort.InferenceSession(onnx_path)
    
    results = []
    for test_input in test_inputs:
        result = session.run(['output'], {'input': test_input})
        results.append(result[0])
    
    return {
        'model_path': onnx_path,
        'input_shape': session.get_inputs()[0].shape,
        'output_shape': session.get_outputs()[0].shape,
        'test_results': results
    }
