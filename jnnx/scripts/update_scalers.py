#!/usr/bin/env python3
"""
Extract scalers from PyTorch .pth files and update the .jnnx package.

This script updates the scalers.pkl file in the .jnnx package with scalers
extracted from PyTorch .pth files.
"""

import json
import sys
import pickle
from pathlib import Path
from datetime import datetime

def extract_scalers_from_pth(pth_file, input_dim, output_dim):
    """
    Extract scaler parameters from a .pth file.
    Returns a dictionary of scaler data or None if extraction fails.
    """
    try:
        import torch
        print(f"Loading {pth_file}...")
        checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)

        x_scaler = checkpoint.get('x_scaler')
        y_scaler = checkpoint.get('y_scaler')

        if x_scaler is None or y_scaler is None:
            raise ValueError("Scalers (x_scaler, y_scaler) not found in .pth file.")

        return {
            "x_min": x_scaler.data_min_.tolist(),
            "x_max": x_scaler.data_max_.tolist(),
            "y_min": y_scaler.data_min_.tolist(),
            "y_max": y_scaler.data_max_.tolist()
        }

    except Exception as e:
        print(f"Error extracting scalers from .pth file: {e}")
        return None


def create_default_scalers(input_dim, output_dim, metadata):
    """Create default scalers matching input limits from metadata."""
    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])

    # Use input parameter limits
    x_min = [param.get('min', 0.0) for param in input_params]
    x_max = [param.get('max', 1.0) for param in input_params]

    # Use output parameter limits
    y_min = [param.get('min', 0.0) for param in output_params]
    y_max = [param.get('max', 1.0) for param in output_params]

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max
    }


def update_jnnx_scalers(jnnx_dir):
    """
    Extract scaler parameters from a .pth file and update the .jnnx package.
    """
    jnnx_path = Path(jnnx_dir)

    # Find required files
    metadata_file = jnnx_path / "metadata.json"
    scalers_file = jnnx_path / "scalers.pkl"

    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {jnnx_path}")
        return False

    # Load metadata to get dimensions
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])
    input_dim = len(input_params)
    output_dim = len(output_params)

    print(f"Updating scalers for: {jnnx_path}")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Output dimensions: {output_dim}")

    # Try to extract scalers from .pth file if it exists
    scalers_data = None
    pth_files = list(jnnx_path.glob("*.pth"))
    if pth_files:
        pth_file = pth_files[0]
        print(f"  Source .pth file: {pth_file}")
        scalers_data = extract_scalers_from_pth(pth_file, input_dim, output_dim)

    if scalers_data is None:
        print("Warning: Could not extract scalers from .pth file, using default values")
        scalers_data = create_default_scalers(input_dim, output_dim, metadata)

    # Save scalers to pickle file
    with open(scalers_file, 'wb') as f:
        pickle.dump(scalers_data, f)

    # Save scalers to text file (as per jnnx-format-spec.md)
    scalers_txt_file = jnnx_path / "scalers.txt"
    with open(scalers_txt_file, 'w') as f:
        # Write in order: x_min, x_max, y_min, y_max (as per spec)
        for value in scalers_data['x_min']:
            f.write(f"{value}\n")
        for value in scalers_data['x_max']:
            f.write(f"{value}\n")
        for value in scalers_data['y_min']:
            f.write(f"{value}\n")
        for value in scalers_data['y_max']:
            f.write(f"{value}\n")

    print(f"✓ Scalers updated successfully in {scalers_file}")
    print(f"✓ Scalers text file created: {scalers_txt_file}")
    print(f"  Input scalers: min={scalers_data['x_min']}, max={scalers_data['x_max']}")
    print(f"  Output scalers: min={scalers_data['y_min']}, max={scalers_data['y_max']}")

    return True


def main():
    """Command line interface for scaler extraction."""
    if len(sys.argv) < 2:
        print("Usage: python update-scalers.py <jnnx_directory>")
        print("       python update-scalers.py models/sdt.jnnx/")
        print("")
        sys.exit(1)

    jnnx_dir = sys.argv[1]
    print(f"Updating scalers for: {jnnx_dir}")
    print("=" * 50)

    if update_jnnx_scalers(jnnx_dir):
        print("\n✓ Scaler update complete!")
        print(f"  Directory: {jnnx_dir}")
        print(f"  Updated: scalers.pkl, scalers.txt")
    else:
        print("\n✗ Scaler update failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
