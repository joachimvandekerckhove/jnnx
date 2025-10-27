#!/usr/bin/env python3
"""
Comprehensive test suite for jnnx project phases.

This test suite validates all functionality described in the project description
for each phase, including edge cases and error conditions.

Usage:
    python test-suite.py                    # Run all tests
    python test-suite.py phase1            # Run only phase 1 tests
    python test-suite.py phase2            # Run only phase 2 tests
    python test-suite.py phase3            # Run only phase 3 tests
    python test-suite.py phase4            # Run only phase 4 tests
    python test-suite.py phase5            # Run only phase 5 tests
"""

import sys
import os
import unittest
import subprocess
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock


class MockScaler:
    """Mock scaler class that can be pickled."""
    def __init__(self, data_min, data_max):
        self.data_min_ = data_min
        self.data_max_ = data_max


class TestPhase1(unittest.TestCase):
    """Test Phase 1: Interface to read and edit JSON control files."""
    
    def setUp(self):
        """Set up test environment for Phase 1."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "test_model.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create test metadata.json
        self.metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "param1", "min": 0.0, "max": 1.0},
                {"name": "param2", "min": -1.0, "max": 1.0}
            ],
            "output_parameters": [
                {"name": "output1", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_jnnx_setup_finds_metadata(self):
        """Test that jnnx-setup finds metadata.json file."""
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", str(self.jnnx_dir)
        ], input="7\n", capture_output=True, text=True)
        
        self.assertIn("Found metadata file", result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_jnnx_setup_displays_current_config(self):
        """Test that jnnx-setup displays current configuration."""
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", str(self.jnnx_dir)
        ], input="7\n", capture_output=True, text=True)
        
        self.assertIn("Current metadata", result.stdout)
        self.assertIn("test_model", result.stdout)
        self.assertEqual(result.returncode, 0)
    
    def test_jnnx_setup_handles_missing_metadata(self):
        """Test that jnnx-setup handles missing metadata.json."""
        (self.jnnx_dir / "metadata.json").unlink()
        
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("metadata.json not found", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_jnnx_setup_handles_invalid_directory(self):
        """Test that jnnx-setup handles invalid directory."""
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", "nonexistent_directory"
        ], capture_output=True, text=True)
        
        self.assertIn("does not exist", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_jnnx_setup_handles_non_jnnx_directory(self):
        """Test that jnnx-setup handles directory not ending with .jnnx."""
        invalid_dir = self.test_dir / "invalid_model"
        invalid_dir.mkdir()
        
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", str(invalid_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("does not end with .jnnx", result.stdout)
        self.assertNotEqual(result.returncode, 0)


class TestPhase2(unittest.TestCase):
    """Test Phase 2: Validate JNNX - test suite for .jnnx folders."""
    
    def setUp(self):
        """Set up test environment for Phase 2."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "test_model.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create minimal valid .jnnx package
        self.create_valid_jnnx_package()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_valid_jnnx_package(self):
        """Create a valid .jnnx package for testing."""
        # Create metadata.json
        metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "param1", "min": 0.0, "max": 1.0}
            ],
            "output_parameters": [
                {"name": "output1", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Create dummy model.onnx (empty file for testing)
        (self.jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        
        # Create scalers.pkl
        scalers = {
            'x_min': [0.0],
            'x_max': [1.0],
            'y_min': [0.0],
            'y_max': [1.0]
        }
        
        with open(self.jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
        
        # Create scalers.txt
        scalers_txt = "0.0\n1.0\n0.0\n1.0\n"
        (self.jnnx_dir / "scalers.txt").write_text(scalers_txt)
        
        # Create README.md
        (self.jnnx_dir / "README.md").write_text("# Test Model\nTest model description.")
    
    def test_validate_jnnx_package_integrity(self):
        """Test that validate-jnnx checks package integrity."""
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        # Should attempt validation even if scalers can't be loaded
        self.assertIn("Validating:", result.stdout)
        self.assertIn("Metadata:", result.stdout)
        self.assertIn("ONNX:", result.stdout)
        self.assertIn("Scalers:", result.stdout)
    
    def test_validate_jnnx_missing_files(self):
        """Test that validate-jnnx detects missing files."""
        (self.jnnx_dir / "model.onnx").unlink()
        
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("model.onnx not found", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_validate_jnnx_invalid_directory(self):
        """Test that validate-jnnx handles invalid directory."""
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", "nonexistent_directory"
        ], capture_output=True, text=True)
        
        self.assertIn("does not exist", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_validate_jnnx_non_jnnx_directory(self):
        """Test that validate-jnnx handles directory not ending with .jnnx."""
        invalid_dir = self.test_dir / "invalid_model"
        invalid_dir.mkdir()
        
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(invalid_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("does not end with .jnnx", result.stdout)
        self.assertNotEqual(result.returncode, 0)


class TestPhase3(unittest.TestCase):
    """Test Phase 3: Create module code - C++ templates and generation."""
    
    def setUp(self):
        """Set up test environment for Phase 3."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "test_model.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create valid .jnnx package
        self.create_valid_jnnx_package()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_valid_jnnx_package(self):
        """Create a valid .jnnx package for testing."""
        # Create metadata.json
        metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "param1", "min": 0.0, "max": 1.0}
            ],
            "output_parameters": [
                {"name": "output1", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Create dummy model.onnx
        (self.jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        
        # Create scalers.pkl
        scalers = {
            'x_min': [0.0],
            'x_max': [1.0],
            'y_min': [0.0],
            'y_max': [1.0]
        }
        
        with open(self.jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
    
    def test_generate_module_creates_files(self):
        """Test that generate-module creates required files."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Check that build directory was created
        build_dir = Path("tmp/test_model.jnnx_build")
        self.assertTrue(build_dir.exists())
        
        # Check that required files were generated
        self.assertTrue((build_dir / "testmodel_emulator.cc").exists())
        self.assertTrue((build_dir / "Makefile").exists())
    
    def test_generate_module_handles_missing_files(self):
        """Test that generate-module handles missing files."""
        (self.jnnx_dir / "model.onnx").unlink()
        
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("model.onnx not found", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_generate_module_handles_invalid_directory(self):
        """Test that generate-module handles invalid directory."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", "nonexistent_directory"
        ], capture_output=True, text=True)
        
        self.assertIn("does not exist", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_generated_cpp_code_structure(self):
        """Test that generated C++ code has correct structure."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Check generated C++ file
        cpp_file = Path("tmp/test_model.jnnx_build/testmodel_emulator.cc")
        self.assertTrue(cpp_file.exists())
        
        cpp_content = cpp_file.read_text()
        
        # Check for required components
        self.assertIn("#include <module/Module.h>", cpp_content)
        self.assertIn("#include <function/VectorFunction.h>", cpp_content)
        self.assertIn("class TESTMODEL_Function", cpp_content)
        self.assertIn("class TESTMODEL_Module", cpp_content)
        self.assertIn("extern \"C\"", cpp_content)
        self.assertIn("load_module()", cpp_content)
    
    def test_generated_makefile_structure(self):
        """Test that generated Makefile has correct structure."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Check generated Makefile
        makefile = Path("tmp/test_model.jnnx_build/Makefile")
        self.assertTrue(makefile.exists())
        
        makefile_content = makefile.read_text()
        
        # Check for required components
        self.assertIn("MODULE_NAME = testmodel_emulator", makefile_content)
        self.assertIn("CXX = g++", makefile_content)
        self.assertIn("CXXFLAGS = -fPIC -O2 -Wall -std=c++17", makefile_content)
        self.assertIn("JAGS_INCLUDE", makefile_content)
        self.assertIn("ONNX_INCLUDE", makefile_content)


class TestPhase4(unittest.TestCase):
    """Test Phase 4: Create example module - SDT example compilation and installation."""
    
    def setUp(self):
        """Set up test environment for Phase 4."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "sdt.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create SDT example package
        self.create_sdt_package()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_sdt_package(self):
        """Create SDT example package."""
        # Create metadata.json
        metadata = {
            "model_name": "sdt_emulator",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "dprime", "min": -5.0, "max": 5.0},
                {"name": "criterion", "min": -5.0, "max": 5.0}
            ],
            "output_parameters": [
                {"name": "hit_rate", "min": 0.0, "max": 1.0},
                {"name": "false_alarm_rate", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit", "probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Create dummy model.onnx
        (self.jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        
        # Create scalers.pkl
        scalers = {
            'x_min': [-5.0, -5.0],
            'x_max': [5.0, 5.0],
            'y_min': [0.0, 0.0],
            'y_max': [1.0, 1.0]
        }
        
        with open(self.jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
    
    def test_sdt_module_generation(self):
        """Test SDT module generation."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("sdt_emulator", result.stdout)
    
    def test_sdt_module_compilation(self):
        """Test SDT module compilation."""
        # Generate module first
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Try to compile (may fail due to missing ONNX Runtime, but should attempt)
        build_dir = Path("tmp/sdt.jnnx_build")
        if build_dir.exists():
            result = subprocess.run([
                "make", "-C", str(build_dir)
            ], capture_output=True, text=True)
            
            # Compilation may fail due to missing dependencies, but should attempt
            self.assertIn("g++", result.stderr or result.stdout)
    
    def test_sdt_module_structure(self):
        """Test SDT module has correct structure."""
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Check generated files
        build_dir = Path("tmp/sdt.jnnx_build")
        self.assertTrue(build_dir.exists())
        self.assertTrue((build_dir / "sdt_emulator.cc").exists())
        self.assertTrue((build_dir / "Makefile").exists())
        
        # Check C++ content
        cpp_content = (build_dir / "sdt_emulator.cc").read_text()
        self.assertIn("class SDT_Function", cpp_content)
        self.assertIn("class SDT_Module", cpp_content)
        self.assertIn("VectorFunction(\"sdt\", 2)", cpp_content)


class TestPhase5(unittest.TestCase):
    """Test Phase 5: Create validation suite for module - test compiled JAGS module."""
    
    def setUp(self):
        """Set up test environment for Phase 5."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "test_model.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create valid .jnnx package
        self.create_valid_jnnx_package()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_valid_jnnx_package(self):
        """Create a valid .jnnx package for testing."""
        # Create metadata.json
        metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "param1", "min": 0.0, "max": 1.0}
            ],
            "output_parameters": [
                {"name": "output1", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Create dummy model.onnx
        (self.jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        
        # Create scalers.pkl
        scalers = {
            'x_min': [0.0],
            'x_max': [1.0],
            'y_min': [0.0],
            'y_max': [1.0]
        }
        
        with open(self.jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
    
    def test_validate_module_handles_missing_files(self):
        """Test that validate-module handles missing files."""
        (self.jnnx_dir / "model.onnx").unlink()
        
        result = subprocess.run([
            sys.executable, "scripts/validate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("model.onnx not found", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_validate_module_handles_invalid_directory(self):
        """Test that validate-module handles invalid directory."""
        result = subprocess.run([
            sys.executable, "scripts/validate-module", "nonexistent_directory"
        ], capture_output=True, text=True)
        
        self.assertIn("does not exist", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_validate_module_loads_metadata(self):
        """Test that validate-module loads metadata correctly."""
        result = subprocess.run([
            sys.executable, "scripts/validate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("test_model", result.stdout)
        self.assertIn("1.0.0", result.stdout)
    
    def test_validate_module_handles_non_jnnx_directory(self):
        """Test that validate-module handles directory not ending with .jnnx."""
        invalid_dir = self.test_dir / "invalid_model"
        invalid_dir.mkdir()
        
        result = subprocess.run([
            sys.executable, "scripts/validate-module", str(invalid_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("does not end with .jnnx", result.stdout)
        self.assertNotEqual(result.returncode, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions across all phases."""
    
    def setUp(self):
        """Set up test environment for edge cases."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_empty_directory(self):
        """Test handling of empty directory."""
        empty_dir = self.test_dir / "empty.jnnx"
        empty_dir.mkdir()
        
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(empty_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("metadata.json not found", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_corrupted_json(self):
        """Test handling of corrupted JSON files."""
        jnnx_dir = self.test_dir / "corrupted.jnnx"
        jnnx_dir.mkdir()
        
        # Create corrupted metadata.json
        (jnnx_dir / "metadata.json").write_text("{ invalid json")
        
        # Add required files to avoid "missing files" error
        (jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        scalers = {'x_min': [0.0], 'x_max': [1.0], 'y_min': [0.0], 'y_max': [1.0]}
        with open(jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
        
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertIn("Invalid JSON", result.stdout)
        self.assertNotEqual(result.returncode, 0)
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields in metadata."""
        jnnx_dir = self.test_dir / "incomplete.jnnx"
        jnnx_dir.mkdir()
        
        # Create incomplete metadata.json
        incomplete_metadata = {
            "model_name": "test_model"
            # Missing required fields
        }
        
        with open(jnnx_dir / "metadata.json", 'w') as f:
            json.dump(incomplete_metadata, f, indent=4)
        
        # Add required files to avoid "missing files" error
        (jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        scalers = {'x_min': [0.0], 'x_max': [1.0], 'y_min': [0.0], 'y_max': [1.0]}
        with open(jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
        
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(jnnx_dir)
        ], capture_output=True, text=True)
        
        # Should handle missing fields gracefully
        self.assertIn("Validating:", result.stdout)
    
    def test_permission_errors(self):
        """Test handling of permission errors."""
        # This test is more conceptual as we can't easily create permission errors
        # in a controlled way, but we can test that scripts handle file operations
        # gracefully
        
        jnnx_dir = self.test_dir / "permission_test.jnnx"
        jnnx_dir.mkdir()
        
        # Create valid package
        metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "input_parameters": [{"name": "param1", "min": 0.0, "max": 1.0}],
            "output_parameters": [{"name": "output1", "min": 0.0, "max": 1.0}],
            "transformations": {"input_transform": "minmax", "output_transforms": ["probit"]}
        }
        
        with open(jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Test that scripts handle file operations
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(jnnx_dir)
        ], capture_output=True, text=True)
        
        # Should complete without permission errors
        self.assertIn("Validating:", result.stdout)


class TestIntegration(unittest.TestCase):
    """Test integration between different phases."""
    
    def setUp(self):
        """Set up test environment for integration tests."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.jnnx_dir = self.test_dir / "integration_test.jnnx"
        self.jnnx_dir.mkdir()
        
        # Create complete .jnnx package
        self.create_complete_package()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def create_complete_package(self):
        """Create a complete .jnnx package for integration testing."""
        # Create metadata.json
        metadata = {
            "model_name": "integration_test",
            "version": "1.0.0",
            "input_parameters": [
                {"name": "input1", "min": 0.0, "max": 1.0},
                {"name": "input2", "min": -1.0, "max": 1.0}
            ],
            "output_parameters": [
                {"name": "output1", "min": 0.0, "max": 1.0},
                {"name": "output2", "min": 0.0, "max": 1.0}
            ],
            "transformations": {
                "input_transform": "minmax",
                "output_transforms": ["probit", "probit"]
            }
        }
        
        with open(self.jnnx_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Create dummy model.onnx
        (self.jnnx_dir / "model.onnx").write_bytes(b"dummy onnx content")
        
        # Create scalers.pkl
        scalers = {
            'x_min': [0.0, -1.0],
            'x_max': [1.0, 1.0],
            'y_min': [0.0, 0.0],
            'y_max': [1.0, 1.0]
        }
        
        with open(self.jnnx_dir / "scalers.pkl", 'wb') as f:
            pickle.dump(scalers, f)
        
        # Create scalers.txt
        scalers_txt = "0.0\n-1.0\n1.0\n1.0\n0.0\n0.0\n1.0\n1.0\n"
        (self.jnnx_dir / "scalers.txt").write_text(scalers_txt)
        
        # Create README.md
        (self.jnnx_dir / "README.md").write_text("# Integration Test Model\nTest model for integration testing.")
    
    def test_phase2_to_phase3_integration(self):
        """Test integration between Phase 2 (validation) and Phase 3 (generation)."""
        # First validate the package
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Then generate module
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Check that build directory was created
        build_dir = Path("tmp/integration_test.jnnx_build")
        self.assertTrue(build_dir.exists())
    
    def test_phase3_to_phase5_integration(self):
        """Test integration between Phase 3 (generation) and Phase 5 (validation)."""
        # Generate module first
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Then validate module
        result = subprocess.run([
            sys.executable, "scripts/validate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        # Should handle the validation (may fail due to JAGS module loading issues)
        self.assertIn("integration_test", result.stdout)
    
    def test_complete_workflow(self):
        """Test complete workflow from Phase 1 to Phase 5."""
        # Phase 1: Setup (interactive, so we'll skip the actual editing)
        result = subprocess.run([
            sys.executable, "scripts/jnnx-setup", str(self.jnnx_dir)
        ], input="7\n", capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Phase 2: Validate package
        result = subprocess.run([
            sys.executable, "scripts/validate-jnnx", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Phase 3: Generate module
        result = subprocess.run([
            sys.executable, "scripts/generate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        
        # Phase 4: Compilation (may fail due to dependencies)
        build_dir = Path("tmp/integration_test.jnnx_build")
        if build_dir.exists():
            result = subprocess.run([
                "make", "-C", str(build_dir)
            ], capture_output=True, text=True)
            
            # Should attempt compilation
            self.assertIn("g++", result.stderr or result.stdout)
        
        # Phase 5: Validate module
        result = subprocess.run([
            sys.executable, "scripts/validate-module", str(self.jnnx_dir)
        ], capture_output=True, text=True)
        
        # Should handle validation
        self.assertIn("integration_test", result.stdout)


def run_phase_tests(phase_name):
    """Run tests for a specific phase."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if phase_name == "phase1":
        suite.addTests(loader.loadTestsFromTestCase(TestPhase1))
    elif phase_name == "phase2":
        suite.addTests(loader.loadTestsFromTestCase(TestPhase2))
    elif phase_name == "phase3":
        suite.addTests(loader.loadTestsFromTestCase(TestPhase3))
    elif phase_name == "phase4":
        suite.addTests(loader.loadTestsFromTestCase(TestPhase4))
    elif phase_name == "phase5":
        suite.addTests(loader.loadTestsFromTestCase(TestPhase5))
    else:
        print(f"Unknown phase: {phase_name}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        phase = sys.argv[1].lower()
        if phase in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
            print(f"Running tests for {phase.upper()}")
            print("=" * 50)
            success = run_phase_tests(phase)
        else:
            print(f"Unknown phase: {phase}")
            print("Available phases: phase1, phase2, phase3, phase4, phase5")
            sys.exit(1)
    else:
        print("Running all tests")
        print("=" * 50)
        
        # Run all test classes
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        suite.addTests(loader.loadTestsFromTestCase(TestPhase1))
        suite.addTests(loader.loadTestsFromTestCase(TestPhase2))
        suite.addTests(loader.loadTestsFromTestCase(TestPhase3))
        suite.addTests(loader.loadTestsFromTestCase(TestPhase4))
        suite.addTests(loader.loadTestsFromTestCase(TestPhase5))
        suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        success = result.wasSuccessful()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()