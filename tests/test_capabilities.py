"""Unit tests for JNNX package capabilities."""

import json
import tempfile
import unittest
from pathlib import Path

from jnnx.capabilities import (
    build_sl_config,
    get_capabilities,
    has_capability,
    validate_capabilities,
    validate_sl_package,
)


class TestCapabilities(unittest.TestCase):
    def test_default_emulator_only(self):
        meta = {"model_name": "sdt"}
        self.assertEqual(get_capabilities(meta), ["emulator"])
        self.assertTrue(has_capability(meta, "emulator"))
        self.assertFalse(has_capability(meta, "synthetic_likelihood"))

    def test_explicit_sl_capabilities(self):
        meta = {
            "capabilities": ["emulator", "synthetic_likelihood"],
            "synthetic_likelihood": {"n_summaries": 3},
        }
        caps = get_capabilities(meta)
        self.assertEqual(caps, ["emulator", "synthetic_likelihood"])
        self.assertTrue(has_capability(meta, "synthetic_likelihood"))

    def test_unknown_capability_rejected(self):
        meta = {"capabilities": ["emulator", "foo"]}
        with self.assertRaises(ValueError):
            get_capabilities(meta)
        errors = validate_capabilities(meta)
        self.assertTrue(any("unknown capability" in e for e in errors))

    def test_emulator_required_in_capabilities(self):
        meta = {"capabilities": ["synthetic_likelihood"]}
        with self.assertRaises(ValueError):
            get_capabilities(meta)

    def test_sl_config_block_required(self):
        meta = {"capabilities": ["emulator", "synthetic_likelihood"]}
        errors = validate_capabilities(meta)
        self.assertTrue(
            any("requires synthetic_likelihood config" in e for e in errors)
        )

    def test_orphan_sl_config_rejected(self):
        meta = {
            "capabilities": ["emulator"],
            "synthetic_likelihood": {"n_summaries": 3},
        }
        errors = validate_capabilities(meta)
        self.assertTrue(any("capability not declared" in e for e in errors))

    def test_legacy_format_version_warns(self):
        meta = {
            "format_version": "1.1.0",
            "capabilities": ["emulator", "synthetic_likelihood"],
            "synthetic_likelihood": {"n_summaries": 3},
        }
        errors = validate_capabilities(meta)
        self.assertTrue(any("format_version is deprecated" in e for e in errors))

    def test_legacy_enabled_flag_warns(self):
        meta = {
            "synthetic_likelihood": {"enabled": True, "n_summaries": 3},
        }
        errors = validate_capabilities(meta)
        self.assertTrue(any("synthetic_likelihood.enabled is deprecated" in e for e in errors))

    def test_validate_sl_package_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            like = {
                "version": "1.0",
                "n_summaries": 3,
                "sigma_emu": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
            (pkg / "likelihood.json").write_text(json.dumps(like))
            (pkg / "model.onnx").write_bytes(b"onnx")

            meta = {
                "capabilities": ["emulator", "synthetic_likelihood"],
                "model_name": "ddm3mv",
                "module_name": "ddm3mv_emulator",
                "function_name": "ddm3mv_emulator",
                "input_parameters": [{"name": "v"}, {"name": "a"}, {"name": "t0"}],
                "output_parameters": [{"name": f"o{i}"} for i in range(9)],
                "synthetic_likelihood": {
                    "n_summaries": 3,
                    "onnx_layout": "concatenated",
                    "distribution_name": "ddm3mv_sl",
                    "trial_count_arg": "n_trials",
                    "include_sigma_emu": True,
                },
            }
            errors = validate_sl_package(meta, pkg)
            self.assertEqual(errors, [])
            cfg = build_sl_config(meta, pkg)
            self.assertEqual(cfg["m_out"], 9)
            self.assertEqual(cfg["distribution_name"], "ddm3mv_sl")


if __name__ == "__main__":
    unittest.main()
