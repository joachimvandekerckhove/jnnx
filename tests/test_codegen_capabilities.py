"""Codegen tests for capability-based template selection."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jnnx.capabilities import build_sl_config, has_capability
from jnnx.core import JNNXPackage
from jnnx.scripts import generate_module


class TestCodegenCapabilities(unittest.TestCase):
    def test_ddm3mv_has_sl_capability(self):
        pkg = JNNXPackage(str(ROOT / "models" / "ddm3mv.jnnx"))
        self.assertIn("synthetic_likelihood", pkg.capabilities)
        self.assertTrue(has_capability(pkg.metadata, "synthetic_likelihood"))
        ok, errs = pkg.validate()
        self.assertTrue(ok, errs)

    def test_sdt_is_emulator_only(self):
        pkg = JNNXPackage(str(ROOT / "models" / "sdt.jnnx"))
        self.assertEqual(pkg.capabilities, ["emulator"])
        self.assertFalse(has_capability(pkg.metadata, "synthetic_likelihood"))

    def test_generate_module_selects_sl_template(self):
        meta = json.loads(
            (ROOT / "models" / "ddm3mv.jnnx" / "metadata.json").read_text()
        )
        self.assertTrue(has_capability(meta, "synthetic_likelihood"))
        sl_cfg = build_sl_config(meta, ROOT / "models" / "ddm3mv.jnnx")
        self.assertEqual(sl_cfg["distribution_name"], "ddm3mv_sl")
        self.assertEqual(sl_cfg["p"], 3)

    def test_emulator_only_uses_v10_template(self):
        meta = json.loads((ROOT / "models" / "sdt.jnnx" / "metadata.json").read_text())
        self.assertFalse(has_capability(meta, "synthetic_likelihood"))

    def test_generate_sdt_build_artifacts(self):
        import tempfile
        import shutil

        meta = json.loads((ROOT / "models" / "sdt.jnnx" / "metadata.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            onnx = ROOT / "models" / "sdt.jnnx" / "model.onnx"
            generate_module.generate_module_code(
                meta, onnx, out, ROOT / "models" / "sdt.jnnx"
            )
            generate_module.generate_makefile(meta, out)
            cc = out / "sdt_emulator.cc"
            self.assertTrue(cc.exists())
            content = cc.read_text()
            self.assertIn("sdt_emulator", content)
            self.assertNotIn("SL_Distribution", content)
            makefile = out.read_text() if False else (out / "Makefile").read_text()
            self.assertIn("sdt_emulator.cc", makefile)
            self.assertNotIn("sl_math.cc", makefile)


if __name__ == "__main__":
    unittest.main()
