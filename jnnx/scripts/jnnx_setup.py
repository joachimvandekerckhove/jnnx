#!/usr/bin/env python3
"""
jnnx-setup: Interface to read and edit metadata.json files for .jnnx packages.

Usage: ./jnnx-setup models/sdt.jnnx/
"""

import json
import sys
from pathlib import Path


def find_metadata_file(jnnx_dir):
    """Find the metadata.json file in a .jnnx directory."""
    jnnx_path = Path(jnnx_dir)
    if not jnnx_path.exists():
        print(f"Error: Directory {jnnx_dir} does not exist")
        sys.exit(1)
    
    if not jnnx_path.name.endswith('.jnnx'):
        print(f"Error: Directory {jnnx_dir} does not end with .jnnx")
        sys.exit(1)
    
    # Look for metadata.json specifically
    metadata_file = jnnx_path / "metadata.json"
    
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {jnnx_dir}")
        print("Expected .jnnx format with metadata.json file")
        sys.exit(1)
    
    return metadata_file


def load_metadata(metadata_file):
    """Load metadata.json configuration."""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {metadata_file}: {e}")
        sys.exit(1)


def save_metadata(metadata_file, config):
    """Save metadata.json configuration."""
    with open(metadata_file, 'w') as f:
        json.dump(config, f, indent=4)


def display_metadata(config):
    """Display current metadata with field clipping."""
    print("\nCurrent metadata:")
    print("=" * 50)

    MAX_FIELD_LENGTH = 75
    
    if not config:
        print("(Empty metadata)")
        return
    
    for key, value in config.items():
        if isinstance(value, str) and len(value) > MAX_FIELD_LENGTH:
            display_value = value[:MAX_FIELD_LENGTH - 3] + "..."
        elif isinstance(value, list) and len(str(value)) > MAX_FIELD_LENGTH:
            display_value = str(value)[:MAX_FIELD_LENGTH - 3] + "..."
        else:
            display_value = value
        
        print(f"{key:20}: {display_value}")


def get_field_value(field_name, current_value=None):
    """Get new value for a field from user input."""
    if current_value is not None:
        print(f"\nCurrent value for '{field_name}': {current_value}")
    
    print(f"Enter new value for '{field_name}' (or press Enter to keep current):")
    new_value = input().strip()
    
    if not new_value and current_value is not None:
        return current_value
    
    # Try to parse as JSON for complex types
    if new_value.startswith('[') or new_value.startswith('{'):
        try:
            return json.loads(new_value)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON format, treating as string")
            return new_value
    
    # Try to parse as number
    try:
        if '.' in new_value:
            return float(new_value)
        else:
            return int(new_value)
    except ValueError:
        return new_value


def edit_metadata(config):
    """Interactive metadata editing."""
    if not config:
        print("\nNo existing metadata. Creating new metadata.")
        config = {
            "model_name": "",
            "version": "1.0.0",
            "input_parameters": [],
            "output_parameters": [],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": []
            }
        }
    
    # Define editable fields based on metadata.json structure
    fields = [
        "model_name",
        "version",
        "input_parameters",
        "output_parameters", 
        "transformations"
    ]
    
    while True:
        print("\n" + "=" * 50)
        print("Available fields to edit:")
        for i, field in enumerate(fields, 1):
            current = config.get(field, "(not set)")
            if isinstance(current, str) and len(current) > 40:
                current = current[:37] + "..."
            elif isinstance(current, list) and len(str(current)) > 40:
                current = str(current)[:37] + "..."
            elif isinstance(current, dict) and len(str(current)) > 40:
                current = str(current)[:37] + "..."
            print(f"{i:2}. {field:20}: {current}")
        
        print(f"{len(fields)+1:2}. Save and exit")
        print(f"{len(fields)+2:2}. Exit without saving")
        
        try:
            choice = int(input("\nSelect field to edit (number): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if choice == len(fields) + 1:
            return config
        elif choice == len(fields) + 2:
            return None
        elif 1 <= choice <= len(fields):
            field_name = fields[choice - 1]
            current_value = config.get(field_name)
            
            if field_name in ["input_parameters", "output_parameters"]:
                print(f"\nEditing {field_name}:")
                print("Format: [{\"name\": \"param_name\", \"min\": 0.0, \"max\": 1.0}, ...]")
                print("Enter JSON array or press Enter to keep current:")
                new_value = input().strip()
                if new_value:
                    try:
                        config[field_name] = json.loads(new_value)
                    except json.JSONDecodeError:
                        print("Warning: Invalid JSON format, keeping current value")
                else:
                    config[field_name] = current_value
            elif field_name == "transformations":
                print(f"\nEditing {field_name}:")
                print("Format: {\"input_transform\": \"minmax\", \"output_transforms\": [\"probit\", \"log\"]}")
                print("Enter JSON object or press Enter to keep current:")
                new_value = input().strip()
                if new_value:
                    try:
                        config[field_name] = json.loads(new_value)
                    except json.JSONDecodeError:
                        print("Warning: Invalid JSON format, keeping current value")
                else:
                    config[field_name] = current_value
            else:
                new_value = get_field_value(field_name, current_value)
                config[field_name] = new_value


def main():
    if len(sys.argv) != 2:
        print("Usage: ./jnnx-setup <jnnx-directory>")
        print("Example: ./jnnx-setup models/sdt.jnnx/")
        sys.exit(1)
    
    jnnx_dir = sys.argv[1]
    
    # Find and load metadata file
    metadata_file = find_metadata_file(jnnx_dir)
    print(f"Found metadata file: {metadata_file}")
    
    config = load_metadata(metadata_file)
    
    # Display current configuration
    display_metadata(config)
    
    # Interactive editing
    print("\n" + "=" * 50)
    print("Interactive metadata editor")
    print("=" * 50)
    
    edited_config = edit_metadata(config)
    
    if edited_config is not None:
        # Save changes
        save_metadata(metadata_file, edited_config)
        print(f"\nMetadata saved to: {metadata_file}")
    else:
        print("\nChanges not saved.")


if __name__ == "__main__":
    main()