#!/usr/bin/env python3
"""
Extract scaler parameters from PyTorch .pth files and save to text files for C++ consumption.
This replaces the previous .pkl-based approach with direct .pth file reading.
"""

import torch
import json
import sys
from pathlib import Path
import argparse

def extract_scalers_from_pth(pth_file, output_file):
    """
    Extract scaler parameters from a .pth file and save to text file.
    
    Args:
        pth_file: Path to the .pth file
        output_file: Path to output text file for C++
    """
    try:
        # Load the .pth file
        checkpoint = torch.load(pth_file, map_location='cpu')
        
        # Extract scalers from the checkpoint
        x_scaler = checkpoint.get('x_scaler')
        y_scaler = checkpoint.get('y_scaler')
        
        if x_scaler is None or y_scaler is None:
            raise ValueError("Scalers not found in .pth file")
        
        # Extract scaling parameters
        x_min = x_scaler.data_min_
        x_max = x_scaler.data_max_
        y_min = y_scaler.data_min_
        y_max = y_scaler.data_max_
        
        # Write to text file in format: x_min, x_max, y_min, y_max
        with open(output_file, 'w') as f:
            # Write x_min values (input parameters)
            for val in x_min:
                f.write(f"{val}\n")
            # Write x_max values (input parameters)
            for val in x_max:
                f.write(f"{val}\n")
            # Write y_min values (output statistics)
            for val in y_min:
                f.write(f"{val}\n")
            # Write y_max values (output statistics)
            for val in y_max:
                f.write(f"{val}\n")
        
        print(f"Scalers extracted successfully to {output_file}")
        print(f"Input dimensions: {len(x_min)}")
        print(f"Output dimensions: {len(y_min)}")
        
    except Exception as e:
        print(f"Error extracting scalers: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Extract scalers from .pth files')
    parser.add_argument('pth_file', help='Path to .pth file')
    parser.add_argument('output_file', help='Path to output text file')
    
    args = parser.parse_args()
    
    pth_path = Path(args.pth_file)
    output_path = Path(args.output_file)
    
    if not pth_path.exists():
        print(f"Error: .pth file {pth_path} does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extract_scalers_from_pth(pth_path, output_path)

if __name__ == "__main__":
    main()
