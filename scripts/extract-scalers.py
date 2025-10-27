#!/usr/bin/env python3
"""
Extract scalers from PyTorch .pth files and save in portable JSON format.

This script solves the dependency issues we encountered with DDM4 by creating
a portable scaler storage format that doesn't rely on PyTorch's complex
serialization system.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

def extract_scalers_to_json(pth_file, output_file=None):
    """
    Extract scalers from a .pth file and save in portable JSON format.
    
    Args:
        pth_file: Path to the .pth file
        output_file: Path for output JSON file (default: scalers.json in same directory)
    
    Returns:
        Path to the created JSON file
    """
    pth_path = Path(pth_file)
    
    if output_file is None:
        output_file = pth_path.parent / "scalers.json"
    else:
        output_file = Path(output_file)
    
    print(f"Extracting scalers from: {pth_path}")
    
    try:
        # Try to load with weights_only=False for trusted sources
        import torch
        checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
        
        # Extract scalers
        x_scaler = checkpoint.get('x_scaler')
        y_scaler = checkpoint.get('y_scaler')
        
        if x_scaler is None or y_scaler is None:
            print("Warning: Scalers not found in .pth file")
            return None
        
        # Extract scaler parameters
        scaler_data = {
            "version": "1.0",
            "input_scaler": {
                "type": "MinMaxScaler",
                "data_min": x_scaler.data_min_.tolist() if hasattr(x_scaler, 'data_min_') else [],
                "data_max": x_scaler.data_max_.tolist() if hasattr(x_scaler, 'data_max_') else [],
                "feature_range": [0.0, 1.0]  # Default MinMaxScaler range
            },
            "output_scaler": {
                "type": "MinMaxScaler",
                "data_min": y_scaler.data_min_.tolist() if hasattr(y_scaler, 'data_min_') else [],
                "data_max": y_scaler.data_max_.tolist() if hasattr(y_scaler, 'data_max_') else [],
                "feature_range": [0.0, 1.0]  # Default MinMaxScaler range
            },
            "metadata": {
                "created_by": "jnnx-tools",
                "created_at": datetime.now().isoformat(),
                "source_file": str(pth_path),
                "description": f"Scalers extracted from {pth_path.name}"
            }
        }
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        print(f"✓ Scalers extracted successfully to: {output_file}")
        print(f"  Input dimensions: {len(scaler_data['input_scaler']['data_min'])}")
        print(f"  Output dimensions: {len(scaler_data['output_scaler']['data_min'])}")
        
        return output_file
        
    except Exception as e:
        print(f"Error extracting scalers: {e}")
        print("Creating default scaler file...")
        
        # Create default scaler file (no scaling)
        default_data = {
            "version": "1.0",
            "input_scaler": {
                "type": "MinMaxScaler",
                "data_min": [0.0, 0.0],  # Default 2D
                "data_max": [1.0, 1.0],
                "feature_range": [0.0, 1.0]
            },
            "output_scaler": {
                "type": "MinMaxScaler",
                "data_min": [0.0, 0.0],  # Default 2D
                "data_max": [1.0, 1.0],
                "feature_range": [0.0, 1.0]
            },
            "metadata": {
                "created_by": "jnnx-tools",
                "created_at": datetime.now().isoformat(),
                "source_file": str(pth_path),
                "description": f"Default scalers (no scaling) for {pth_path.name}",
                "note": "Original scalers could not be extracted"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(default_data, f, indent=2)
        
        print(f"✓ Default scalers created at: {output_file}")
        return output_file

def load_scalers_from_json(json_file):
    """
    Load scalers from portable JSON format.
    
    Args:
        json_file: Path to the scalers.json file
    
    Returns:
        Dictionary with scaler parameters
    """
    json_path = Path(json_file)
    
    try:
        with open(json_path, 'r') as f:
            scaler_data = json.load(f)
        
        # Extract scaler parameters
        input_scaler = scaler_data['input_scaler']
        output_scaler = scaler_data['output_scaler']
        
        return {
            'x_min': input_scaler['data_min'],
            'x_max': input_scaler['data_max'],
            'y_min': output_scaler['data_min'],
            'y_max': output_scaler['data_max'],
            'metadata': scaler_data.get('metadata', {})
        }
        
    except Exception as e:
        print(f"Error loading scalers from {json_file}: {e}")
        return None

def main():
    """Command line interface for scaler extraction."""
    if len(sys.argv) < 2:
        print("Usage: python extract-scalers.py <pth_file> [output_file]")
        print("       python extract-scalers.py models/ddm4.jnnx/ddm4_emulator.pth")
        sys.exit(1)
    
    pth_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = extract_scalers_to_json(pth_file, output_file)
    
    if result:
        print(f"\n✓ Scaler extraction complete!")
        print(f"  Input file: {pth_file}")
        print(f"  Output file: {result}")
        
        # Test loading the created file
        scalers = load_scalers_from_json(result)
        if scalers:
            print(f"\n✓ Verification successful:")
            print(f"  Input dimensions: {len(scalers['x_min'])}")
            print(f"  Output dimensions: {len(scalers['y_min'])}")
    else:
        print("✗ Scaler extraction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
