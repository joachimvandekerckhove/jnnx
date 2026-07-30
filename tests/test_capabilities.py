"""Unit tests for JNNX package capabilities."""

import json
import tempfile
import unittest
from pathlib import Path

from jnnx.capabilities import (
    KNOWN_TRANSFORMS,
    build_sl_config,
    get_capabilities,
    has_capability,
    load_obs_transform,
    transform_name_to_code,
    validate_capabilities,
    validate_obs_transform,
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
            obs = {
                "version": "1.0",
                "summary_names": ["acc", "rt_mean", "rt_var"],
                "column_transforms": ["identity", "log1p", "log1p"],
                "scaler_mean": [0.8, 0.4, -2.0],
                "scaler_scale": [0.1, 0.2, 0.7],
            }
            (pkg / "likelihood.json").write_text(json.dumps(like))
            (pkg / "obs_transform.json").write_text(json.dumps(obs))
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
            self.assertEqual(cfg["col_transform_codes"], [0, 1, 1])
            self.assertEqual(len(cfg["obs_scaler_mean"]), 3)

    def test_sl_package_requires_obs_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / "likelihood.json").write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "n_summaries": 3,
                        "sigma_emu": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    }
                )
            )
            (pkg / "model.onnx").write_bytes(b"onnx")
            meta = {
                "capabilities": ["emulator", "synthetic_likelihood"],
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
            self.assertTrue(any("obs_transform.json required" in e for e in errors))

    def test_validate_obs_transform_unknown_name(self):
        obs = {
            "version": "1.0",
            "summary_names": ["a", "b"],
            "column_transforms": ["identity", "exp"],
            "scaler_mean": [0.0, 0.0],
            "scaler_scale": [1.0, 1.0],
        }
        errors = validate_obs_transform(obs, n_summaries=2)
        self.assertTrue(any("unknown transform" in e for e in errors))

    def test_transform_name_to_code(self):
        self.assertEqual(transform_name_to_code("log1p"), KNOWN_TRANSFORMS["log1p"])
        with self.assertRaises(ValueError):
            transform_name_to_code("exp")

    def test_load_obs_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            payload = {
                "version": "1.0",
                "summary_names": ["acc"],
                "column_transforms": ["identity"],
                "scaler_mean": [0.5],
                "scaler_scale": [0.1],
            }
            (pkg / "obs_transform.json").write_text(json.dumps(payload))
            loaded = load_obs_transform(pkg)
            self.assertEqual(loaded["summary_names"], ["acc"])


if __name__ == "__main__":
    unittest.main()
