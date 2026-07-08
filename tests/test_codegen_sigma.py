"""Tests for sigma_emu codegen precision and build manifest."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jnnx.capabilities import build_sl_config
from jnnx.scripts import generate_module
from jnnx.sl_sigma import load_sigma_emu_baked, load_sigma_emu_from_likelihood


class TestCodegenSigma(unittest.TestCase):
    def test_double_format_uses_full_precision(self):
        vals = [3.061417492746862e-05, -1.234567890123456e-06]
        formatted = generate_module.format_array(vals, use_double=True)
        self.assertIn("3.0614174927468617e-05", formatted)
        self.assertNotIn("0.000031", formatted)

    def test_build_manifest_records_sigma_emu_baked(self):
        pkg_dir = ROOT / "models" / "ddm3mv.jnnx"
        meta = json.loads((pkg_dir / "metadata.json").read_text())
        sl_cfg = build_sl_config(meta, pkg_dir)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_module.generate_module_code(meta, pkg_dir / "model.onnx", out, pkg_dir)
            manifest = json.loads((out / "build_manifest.json").read_text())
            self.assertIn("sigma_emu_baked", manifest)
            self.assertEqual(manifest["sl_p"], 3)
            baked = load_sigma_emu_baked(out)
            full = load_sigma_emu_from_likelihood(pkg_dir)
            self.assertTrue((baked == full).all())

    def test_generated_cc_contains_full_precision_sigma(self):
        pkg_dir = ROOT / "models" / "ddm3mv.jnnx"
        meta = json.loads((pkg_dir / "metadata.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            generate_module.generate_module_code(
                meta, pkg_dir / "model.onnx", out, pkg_dir
            )
            cc = (out / "ddm3mv_emulator.cc").read_text()
            self.assertIn("kSigmaEmu", cc)
            self.assertNotRegex(cc, r"0\.000031")


if __name__ == "__main__":
    unittest.main()
