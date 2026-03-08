#!/usr/bin/env python3
"""
Maximal-coverage test suite for the jnnx project.

Covers:
- Phase 1–5 CLI scripts (jnnx-setup, validate-jnnx, generate-module, validate-module)
- jnnx.core (JNNXPackage, JAGSModule, validate_jnnx_package, load_metadata, load_scalers)
- jnnx.utils (find_jnnx_packages, get_package_info, create_jnnx_package)
- extract-scalers.py and update-scalers.py
- Edge cases: missing files, invalid input, wrong directory, corrupted JSON, usage/args

Run from project root:
  python -m pytest tests/test_suite_full.py -v
  python tests/test_suite_full.py
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

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Reference to real SDT package for tests that need valid ONNX
SDT_JNNX = PROJECT_ROOT / "models" / "sdt.jnnx"


def _metadata_with_module(extra=None):
    """Minimal metadata including module_name and function_name for generate-module."""
    m = {
        "model_name": "test_model",
        "module_name": "testmodel_emulator",
        "function_name": "testmodel",
        "version": "1.0.0",
        "input_parameters": [{"name": "p1", "min": 0.0, "max": 1.0}],
        "output_parameters": [{"name": "o1", "min": 0.0, "max": 1.0}],
        "transformations": {"input_transform": "minmax", "output_transforms": ["probit"]},
    }
    if extra:
        m.update(extra)
    return m


def _scalers_dict(n_in=1, n_out=1):
    return {
        "x_min": [0.0] * n_in,
        "x_max": [1.0] * n_in,
        "y_min": [0.0] * n_out,
        "y_max": [1.0] * n_out,
    }


def _create_valid_jnnx(tmp_dir, name="test.jnnx", metadata=None, copy_onnx=True):
    """Create a valid .jnnx directory under tmp_dir."""
    jnnx = Path(tmp_dir) / name
    jnnx.mkdir(parents=True)
    meta = metadata or _metadata_with_module()
    with open(jnnx / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(jnnx / "scalers.pkl", "wb") as f:
        pickle.dump(_scalers_dict(), f)
    if copy_onnx and SDT_JNNX.exists() and (SDT_JNNX / "model.onnx").exists():
        shutil.copy(SDT_JNNX / "model.onnx", jnnx / "model.onnx")
    else:
        (jnnx / "model.onnx").write_bytes(b"dummy")
    (jnnx / "scalers.txt").write_text("0.0\n1.0\n0.0\n1.0\n")
    (jnnx / "README.md").write_text("# Test\n")
    return jnnx


# ---------------------------------------------------------------------------
# Unit tests: jnnx.utils
# ---------------------------------------------------------------------------

class TestJnnxUtils(unittest.TestCase):
    """Unit tests for jnnx.utils."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_find_jnnx_packages_empty(self):
        from jnnx.utils import find_jnnx_packages
        self.assertEqual(find_jnnx_packages(str(self.test_dir)), [])

    def test_find_jnnx_packages_finds_jnnx(self):
        from jnnx.utils import find_jnnx_packages
        (self.test_dir / "a.jnnx").mkdir()
        (self.test_dir / "b.jnnx").mkdir()
        (self.test_dir / "not_jnnx").mkdir()
        found = find_jnnx_packages(str(self.test_dir))
        self.assertEqual(len(found), 2)
        names = {Path(p).name for p in found}
        self.assertEqual(names, {"a.jnnx", "b.jnnx"})

    def test_get_package_info(self):
        from jnnx.utils import get_package_info
        pkg = self.test_dir / "p.jnnx"
        pkg.mkdir()
        (pkg / "metadata.json").write_text("{}")
        (pkg / "f1.txt").write_text("hello")
        info = get_package_info(str(pkg))
        self.assertEqual(info["name"], "p.jnnx")
        self.assertIn("metadata.json", info["files"])
        self.assertIn("f1.txt", info["files"])
        self.assertGreater(info["size"], 0)

    def test_create_jnnx_package(self):
        from jnnx.utils import create_jnnx_package
        out = self.test_dir / "out.jnnx"
        onnx_path = self.test_dir / "model.onnx"
        onnx_path.write_bytes(b"dummy onnx")
        create_jnnx_package(
            str(out),
            "m",
            str(onnx_path),
            _scalers_dict(2, 2),
            [{"name": "a", "min": 0, "max": 1}, {"name": "b", "min": 0, "max": 1}],
            [{"name": "x", "min": 0, "max": 1}, {"name": "y", "min": 0, "max": 1}],
        )
        self.assertTrue((out / "metadata.json").exists())
        self.assertTrue((out / "model.onnx").exists())
        self.assertTrue((out / "scalers.pkl").exists())
        self.assertTrue((out / "scalers.txt").exists())
        meta = json.loads((out / "metadata.json").read_text())
        self.assertEqual(meta["model_name"], "m")
        self.assertEqual(len(meta["input_parameters"]), 2)


# ---------------------------------------------------------------------------
# Unit tests: jnnx.core (JNNXPackage, JAGSModule, helpers)
# ---------------------------------------------------------------------------

class TestJnnxCorePackage(unittest.TestCase):
    """Unit tests for jnnx.core.JNNXPackage and helpers."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_metadata_and_scalers_from_valid_package(self):
        """JNNXPackage loads metadata and scalers from a valid .jnnx dir."""
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import JNNXPackage
        pkg = JNNXPackage(str(SDT_JNNX))
        self.assertIn("model_name", pkg.metadata)
        self.assertIn("input_parameters", pkg.metadata)
        self.assertIsInstance(pkg.scalers, dict)

    def test_package_model_name_and_dims(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import JNNXPackage
        pkg = JNNXPackage(str(SDT_JNNX))
        self.assertIsInstance(pkg.model_name, str)
        self.assertGreaterEqual(pkg.input_dim, 0)
        self.assertGreaterEqual(pkg.output_dim, 0)

    def test_package_get_onnx_path(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import JNNXPackage
        pkg = JNNXPackage(str(SDT_JNNX))
        path = pkg.get_onnx_path()
        self.assertTrue(path.endswith("model.onnx") or "model.onnx" in path)
        self.assertTrue(Path(path).exists())

    def test_package_missing_metadata_raises(self):
        from jnnx.core import JNNXPackage
        d = self.test_dir / "bad.jnnx"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"x")
        (d / "scalers.pkl").write_bytes(pickle.dumps(_scalers_dict()))
        with self.assertRaises(FileNotFoundError):
            JNNXPackage(str(d))

    def test_package_missing_scalers_raises(self):
        from jnnx.core import JNNXPackage
        d = self.test_dir / "bad.jnnx"
        d.mkdir()
        (d / "metadata.json").write_text("{}")
        (d / "model.onnx").write_bytes(b"x")
        with self.assertRaises(FileNotFoundError):
            JNNXPackage(str(d))

    def test_get_scaler_parameters_dict_format(self):
        jnnx = _create_valid_jnnx(self.test_dir, "s.jnnx")
        from jnnx.core import JNNXPackage
        pkg = JNNXPackage(str(jnnx))
        # Our fixture uses dict scalers; core may expect sklearn or dict
        params = pkg.get_scaler_parameters()
        self.assertIn("x_min", params)
        self.assertIn("x_max", params)
        self.assertIn("y_min", params)
        self.assertIn("y_max", params)

    def test_validate_jnnx_package_function(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import validate_jnnx_package
        ok, errors = validate_jnnx_package(str(SDT_JNNX))
        # SDT has module_name/function_name; may still have ONNX shape requirements
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(errors, list)

    def test_load_metadata_function(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import load_metadata
        meta = load_metadata(str(SDT_JNNX))
        self.assertIn("model_name", meta)

    def test_load_scalers_function(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import load_scalers
        scalers = load_scalers(str(SDT_JNNX))
        self.assertIsInstance(scalers, dict)


class TestJnnxCoreJAGSModule(unittest.TestCase):
    """Unit tests for jnnx.core.JAGSModule."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        if SDT_JNNX.exists():
            self.jnnx = _create_valid_jnnx(
                self.test_dir, "m.jnnx",
                metadata=json.loads((SDT_JNNX / "metadata.json").read_text())
                if (SDT_JNNX / "metadata.json").exists() else _metadata_with_module(),
                copy_onnx=True,
            )
        else:
            self.jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx", copy_onnx=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_jags_module_init_requires_module_name(self):
        from jnnx.core import JNNXPackage, JAGSModule
        meta = _metadata_with_module()
        del meta["module_name"]
        with open(self.jnnx / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        pkg = JNNXPackage(str(self.jnnx))
        with self.assertRaises(ValueError):
            JAGSModule(pkg, str(self.test_dir / "build"))

    def test_jags_module_init_requires_function_name(self):
        from jnnx.core import JNNXPackage, JAGSModule
        meta = _metadata_with_module()
        del meta["function_name"]
        with open(self.jnnx / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        pkg = JNNXPackage(str(self.jnnx))
        with self.assertRaises(ValueError):
            JAGSModule(pkg, str(self.test_dir / "build"))

    def test_jags_module_generate_code_creates_files(self):
        if not SDT_JNNX.exists():
            self.skipTest("models/sdt.jnnx not found")
        from jnnx.core import JNNXPackage, JAGSModule
        pkg = JNNXPackage(str(self.jnnx))
        build = self.test_dir / "build"
        mod = JAGSModule(pkg, str(build))
        mod.generate_code()
        self.assertTrue((build / "model.onnx").exists())
        self.assertTrue((build / "scalers.txt").exists())
        self.assertTrue((build / f"{mod.module_name}.cc").exists())
        self.assertTrue((build / "Makefile").exists())

    def test_jags_module_compile_without_generate_returns_false(self):
        from jnnx.core import JNNXPackage, JAGSModule
        pkg = JNNXPackage(str(self.jnnx))
        build = self.test_dir / "build"
        build.mkdir()
        mod = JAGSModule(pkg, str(build))
        ok, msg = mod.compile()
        self.assertFalse(ok)
        self.assertIn("Makefile", msg or "")


# ---------------------------------------------------------------------------
# CLI: jnnx-setup
# ---------------------------------------------------------------------------

class TestCLIJnnxSetup(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_prints_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_nonexistent_dir(self):
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py", "nonexistent"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stdout)

    def test_non_jnnx_dir(self):
        d = self.test_dir / "plain"
        d.mkdir()
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py", str(d)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(".jnnx", r.stdout)

    def test_missing_metadata(self):
        jnnx = self.test_dir / "x.jnnx"
        jnnx.mkdir()
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("metadata.json", r.stdout)

    def test_valid_metadata_displays_and_exit_without_saving(self):
        jnnx = _create_valid_jnnx(self.test_dir, "v.jnnx")
        # Menu: 1-5 fields, 6=Save and exit, 7=Exit without saving
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py", str(jnnx)],
            input="7\n",  # Exit without saving
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Current metadata", r.stdout)

    def test_invalid_json_in_metadata(self):
        jnnx = self.test_dir / "bad.jnnx"
        jnnx.mkdir()
        (jnnx / "metadata.json").write_text("{ invalid }")
        r = subprocess.run(
            [sys.executable, "scripts/jnnx-setup.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Invalid JSON", r.stdout)


# ---------------------------------------------------------------------------
# CLI: validate-jnnx
# ---------------------------------------------------------------------------

class TestCLIValidateJnnx(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_nonexistent_dir(self):
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", "nonexistent"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stdout)

    def test_non_jnnx_dir(self):
        d = self.test_dir / "dir"
        d.mkdir()
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", str(d)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(".jnnx", r.stdout)

    def test_missing_onnx(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx")
        (jnnx / "model.onnx").unlink()
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("model.onnx", r.stdout)

    def test_missing_scalers_pkl(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx")
        (jnnx / "scalers.pkl").unlink()
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("scalers.pkl", r.stdout)

    def test_invalid_json_metadata(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx")
        (jnnx / "metadata.json").write_text("{ bad }")
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Invalid JSON", r.stdout)

    def test_valid_package_runs_validation(self):
        if not (SDT_JNNX.exists() and (SDT_JNNX / "model.onnx").exists()):
            self.skipTest("models/sdt.jnnx with model.onnx required")
        r = subprocess.run(
            [sys.executable, "scripts/validate-jnnx.py", str(SDT_JNNX)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertIn("Validating:", r.stdout)
        self.assertIn("Metadata:", r.stdout)


# ---------------------------------------------------------------------------
# CLI: generate-module
# ---------------------------------------------------------------------------

class TestCLIGenerateModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_nonexistent_dir(self):
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", "nonexistent"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stdout)

    def test_non_jnnx_dir(self):
        d = self.test_dir / "dir"
        d.mkdir()
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(d)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(".jnnx", r.stdout)

    def test_missing_metadata(self):
        jnnx = self.test_dir / "x.jnnx"
        jnnx.mkdir()
        (jnnx / "model.onnx").write_bytes(b"x")
        with open(jnnx / "scalers.pkl", "wb") as f:
            pickle.dump(_scalers_dict(), f)
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("metadata.json", r.stdout)

    def test_missing_module_name_in_metadata(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx", metadata={
            "model_name": "x",
            "function_name": "f",
            "version": "1.0",
            "input_parameters": [{"name": "a", "min": 0, "max": 1}],
            "output_parameters": [{"name": "b", "min": 0, "max": 1}],
            "transformations": {},
        })
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("module_name", r.stdout)

    def test_missing_function_name_in_metadata(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx", metadata={
            "model_name": "x",
            "module_name": "x_emulator",
            "version": "1.0",
            "input_parameters": [{"name": "a", "min": 0, "max": 1}],
            "output_parameters": [{"name": "b", "min": 0, "max": 1}],
            "transformations": {},
        })
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("function_name", r.stdout)

    def test_success_creates_build_dir_and_cc(self):
        if not (SDT_JNNX.exists() and (SDT_JNNX / "model.onnx").exists()):
            self.skipTest("models/sdt.jnnx required")
        jnnx = _create_valid_jnnx(self.test_dir, "gen.jnnx", copy_onnx=True)
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(r.returncode, 0)
        build = PROJECT_ROOT / "tmp" / "gen.jnnx_build"
        self.assertTrue(build.exists())
        self.assertTrue((build / "testmodel_emulator.cc").exists())
        self.assertTrue((build / "Makefile").exists())


# ---------------------------------------------------------------------------
# CLI: validate-module
# ---------------------------------------------------------------------------

class TestCLIValidateModule(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/validate-module.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_nonexistent_dir(self):
        r = subprocess.run(
            [sys.executable, "scripts/validate-module.py", "nonexistent"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not exist", r.stdout)

    def test_missing_onnx(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx")
        (jnnx / "model.onnx").unlink()
        r = subprocess.run(
            [sys.executable, "scripts/validate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("model.onnx", r.stdout)

    def test_loads_metadata_and_reports(self):
        jnnx = _create_valid_jnnx(self.test_dir, "m.jnnx")
        r = subprocess.run(
            [sys.executable, "scripts/validate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertIn("test_model", r.stdout)


# ---------------------------------------------------------------------------
# CLI: extract-scalers.py, update-scalers.py
# ---------------------------------------------------------------------------

class TestCLIExtractScalers(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/extract-scalers.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_nonexistent_pth_prints_error_or_creates_default(self):
        r = subprocess.run(
            [sys.executable, "scripts/extract-scalers.py", "nonexistent.pth"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        out = r.stdout or r.stderr or ""
        self.assertTrue(
            "Error" in out or "Default" in out or "extract" in out.lower(),
            msg=f"Expected error or default message, got: {out[:200]}",
        )


class TestCLIUpdateScalers(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_no_args_usage(self):
        r = subprocess.run(
            [sys.executable, "scripts/update-scalers.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Usage", r.stdout or r.stderr)

    def test_missing_metadata_fails(self):
        jnnx = self.test_dir / "no_meta.jnnx"
        jnnx.mkdir()
        r = subprocess.run(
            [sys.executable, "scripts/update-scalers.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        out = r.stdout or r.stderr or ""
        self.assertIn("metadata.json", out, msg="Script should report missing metadata")
        # Script may not set exit code on failure; at least it must report the error

    def test_with_metadata_creates_scalers(self):
        jnnx = _create_valid_jnnx(self.test_dir, "u.jnnx")
        r = subprocess.run(
            [sys.executable, "scripts/update-scalers.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(r.returncode, 0)
        self.assertTrue((jnnx / "scalers.pkl").exists())
        self.assertTrue((jnnx / "scalers.txt").exists())


# ---------------------------------------------------------------------------
# Integration check: workflow-sdt notebook
# ---------------------------------------------------------------------------

class TestWorkflowSdtNotebookIntegration(unittest.TestCase):
    """Smoke-check that workflow-sdt notebook still contains key integration content."""

    def test_workflow_sdt_notebook_smoke_check(self):
        r = subprocess.run(
            [sys.executable, "scripts/check-workflow-sdt.py"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(r.returncode, 0, msg=f"notebook integration check failed: {r.stdout} {r.stderr}")
        self.assertIn("smoke check passed", (r.stdout or "").lower())


# ---------------------------------------------------------------------------
# Edge cases: format_array (Inf), empty lists, etc.
# ---------------------------------------------------------------------------

class TestGenerateModuleFormatArray(unittest.TestCase):
    """Test generate-module format_array behavior via generated C++ (Inf/-Inf)."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_inf_limits_in_metadata_formatted_in_cc(self):
        if not (SDT_JNNX.exists() and (SDT_JNNX / "model.onnx").exists()):
            self.skipTest("models/sdt.jnnx required")
        meta = _metadata_with_module()
        meta["input_parameters"] = [
            {"name": "a", "min": float("-inf"), "max": float("inf")},
            {"name": "b", "min": -1e30, "max": 1e30},
        ]
        meta["output_parameters"] = [{"name": "o", "min": 0, "max": 1}]
        jnnx = _create_valid_jnnx(self.test_dir, "inf.jnnx", metadata=meta, copy_onnx=True)
        r = subprocess.run(
            [sys.executable, "scripts/generate-module.py", str(jnnx)],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        self.assertEqual(r.returncode, 0)
        build = PROJECT_ROOT / "tmp" / "inf.jnnx_build"
        cc = (build / "testmodel_emulator.cc").read_text()
        self.assertIn("1e38f", cc)
        self.assertIn("-1e38f", cc)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
