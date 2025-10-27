#!/usr/bin/env python3
"""
Extract scalers from PyTorch .pth files and update the main JSON configuration file.

This script updates the scalers section in the main JSON configuration file,
keeping everything in one place instead of separate files.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def update_scalers_in_json(jnnx_dir):
    """
    Extract scalers from .pth file and update the main JSON configuration.
    
    Args:
        jnnx_dir: Path to the .jnnx directory
    
    Returns:
        True if successful, False otherwise
    """
    jnnx_path = Path(jnnx_dir)
    
    # Find JSON and PTH files
    json_files = list(jnnx_path.glob("*.json"))
    pth_files = list(jnnx_path.glob("*.pth"))
    
    if not json_files:
        print(f"Error: No JSON file found in {jnnx_path}")
        return False
    
    if not pth_files:
        print(f"Error: No .pth file found in {jnnx_path}")
        return False
    
    # Prioritize main configuration file (not scalers.json)
    main_json_files = [f for f in json_files if f.name != "scalers.json"]
    if main_json_files:
        json_file = main_json_files[0]
    else:
        json_file = json_files[0]  # Fallback to any JSON file
    
    pth_file = pth_files[0]    # Use first .pth file found
    
    print(f"Updating scalers in: {json_file}")
    print(f"Source .pth file: {pth_file}")
    
    # Load existing JSON configuration
    try:
        with open(json_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    # Extract scalers from .pth file
    scalers_data = extract_scalers_from_pth(pth_file)
    
    if scalers_data is None:
        print("Warning: Could not extract scalers, using default values")
        # Create default scalers based on JSON dimensions
        input_dim = config.get('input_dimensions', 2)
        output_dim = config.get('output_dimensions', 2)
        scalers_data = create_default_scalers(input_dim, output_dim, pth_file, config)
    
    # Update the scalers section in the configuration
    config['scalers'] = scalers_data
    
    # Save updated configuration
    try:
        with open(json_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Scalers updated successfully in {json_file}")
        print(f"  Input dimensions: {len(scalers_data['input_scaler']['data_min'])}")
        print(f"  Output dimensions: {len(scalers_data['output_scaler']['data_min'])}")
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False

def extract_scalers_from_pth(pth_file):
    """Extract scalers from .pth file."""
    try:
        import torch
        
        print(f"Loading {pth_file}...")
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
        
        # Extract scalers
        x_scaler = checkpoint.get('x_scaler')
        y_scaler = checkpoint.get('y_scaler')
        
        if x_scaler is None or y_scaler is None:
            print("Warning: Scalers not found in .pth file")
            return None
        
        # Extract scaler parameters
        scalers_data = {
            "input_scaler": {
                "type": "MinMaxScaler",
                "data_min": x_scaler.data_min_.tolist() if hasattr(x_scaler, 'data_min_') else [],
                "data_max": x_scaler.data_max_.tolist() if hasattr(x_scaler, 'data_max_') else [],
                "feature_range": [0.0, 1.0]
            },
            "output_scaler": {
                "type": "MinMaxScaler",
                "data_min": y_scaler.data_min_.tolist() if hasattr(y_scaler, 'data_min_') else [],
                "data_max": y_scaler.data_max_.tolist() if hasattr(y_scaler, 'data_max_') else [],
                "feature_range": [0.0, 1.0]
            },
            "metadata": {
                "created_by": "jnnx-tools",
                "created_at": datetime.now().isoformat(),
                "source_file": str(pth_file),
                "description": f"Scalers extracted from {pth_file.name}"
            }
        }
        
        print("✓ Scalers extracted successfully from .pth file")
        return scalers_data
        
    except Exception as e:
        print(f"Error extracting scalers from .pth file: {e}")
        return None

def create_default_scalers(input_dim, output_dim, pth_file, config):
    """Create default scalers matching input limits."""
    # Use input limits if available, otherwise use [0,1] range
    input_min = config.get('input_limits_min', [0.0] * input_dim)
    input_max = config.get('input_limits_max', [1.0] * input_dim)
    
    # Parse string representations if needed
    if isinstance(input_min, str):
        try:
            input_min = json.loads(input_min)
        except:
            input_min = [0.0] * input_dim
    
    if isinstance(input_max, str):
        try:
            input_max = json.loads(input_max)
        except:
            input_max = [1.0] * input_dim
    
    # Ensure correct dimensions
    if len(input_min) != input_dim:
        input_min = [0.0] * input_dim
    if len(input_max) != input_dim:
        input_max = [1.0] * input_dim
    
    return {
        "input_scaler": {
            "type": "MinMaxScaler",
            "data_min": input_min,
            "data_max": input_max,
            "feature_range": [0.0, 1.0]
        },
        "output_scaler": {
            "type": "MinMaxScaler",
            "data_min": [0.0] * output_dim,
            "data_max": [1.0] * output_dim,
            "feature_range": [0.0, 1.0]
        },
        "metadata": {
            "created_by": "jnnx-tools",
            "created_at": datetime.now().isoformat(),
            "source_file": str(pth_file),
            "description": f"Default scalers matching input limits for {pth_file.name}",
            "note": "Original scalers could not be extracted, using input_limits_min/max"
        }
    }

def main():
    """Command line interface for scaler extraction."""
    if len(sys.argv) < 2:
        print("Usage: python update-scalers.py <jnnx_directory>")
        print("       python update-scalers.py models/ddm4.jnnx/")
        print("")
        print("This script extracts scalers from the .pth file in the directory")
        print("and updates the scalers section in the main JSON configuration file.")
        sys.exit(1)
    
    jnnx_dir = sys.argv[1]
    
    print(f"Updating scalers for: {jnnx_dir}")
    print("=" * 50)
    
    success = update_scalers_in_json(jnnx_dir)
    
    if success:
        print("\n✓ Scaler update complete!")
        print(f"  Directory: {jnnx_dir}")
        print("  Updated: Main JSON configuration file")
    else:
        print("\n✗ Scaler update failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
