"""Tests for JNNX v1.1 synthetic likelihood support."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jnnx.core import JNNXPackage
from jnnx.sl_config import is_sl_package, build_sl_config
from jnnx.scripts.generate_module import is_sl_package as gen_is_sl


class TestV11SL(unittest.TestCase):
    def test_ddm3mv_is_sl_package(self):
        pkg = JNNXPackage(str(ROOT / "models" / "ddm3mv.jnnx"))
        self.assertTrue(pkg.is_sl_package())
        self.assertTrue(is_sl_package(pkg.metadata))
        ok, errs = pkg.validate()
        self.assertTrue(ok, errs)

    def test_sdt_is_not_sl_package(self):
        pkg = JNNXPackage(str(ROOT / "models" / "sdt.jnnx"))
        self.assertFalse(pkg.is_sl_package())

    def test_generate_module_selects_sl_template(self):
        meta = json.loads(
            (ROOT / "models" / "ddm3mv.jnnx" / "metadata.json").read_text()
        )
        self.assertTrue(gen_is_sl(meta))
        sl_cfg = build_sl_config(meta, ROOT / "models" / "ddm3mv.jnnx")
        self.assertEqual(sl_cfg["distribution_name"], "ddm3mv_sl")
        self.assertEqual(sl_cfg["p"], 3)

    def test_v10_template_unchanged(self):
        meta = json.loads((ROOT / "models" / "sdt.jnnx" / "metadata.json").read_text())
        self.assertFalse(gen_is_sl(meta))


if __name__ == "__main__":
    unittest.main()
