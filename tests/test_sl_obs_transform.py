"""Unit tests for observation-space transform reference functions."""

import unittest

import numpy as np

from jnnx.sl_reference import obs_raw_to_std, obs_std_to_raw


class TestObsTransform(unittest.TestCase):
    def test_mixed_roundtrip(self):
        transforms = ["identity", "log1p", "log1p"]
        mean = [0.8, 0.4, -2.0]
        scale = [0.1, 0.2, 0.7]
        raw = [0.75, 0.5, 0.25]
        std, ok = obs_raw_to_std(raw, transforms, mean, scale)
        self.assertTrue(ok)
        back, ok2 = obs_std_to_raw(std, transforms, mean, scale)
        self.assertTrue(ok2)
        np.testing.assert_allclose(back, raw, rtol=0, atol=1e-12)

    def test_identity_only(self):
        transforms = ["identity", "identity"]
        mean = [1.0, -0.5]
        scale = [2.0, 0.5]
        raw = [3.0, 0.0]
        std, ok = obs_raw_to_std(raw, transforms, mean, scale)
        self.assertTrue(ok)
        self.assertAlmostEqual(std[0], (3.0 - 1.0) / 2.0)
        self.assertAlmostEqual(std[1], (0.0 - (-0.5)) / 0.5)

    def test_log_rejects_non_positive(self):
        std, ok = obs_raw_to_std(
            [-1.0], ["log"], [0.0], [1.0]
        )
        self.assertFalse(ok)

    def test_sqrt_roundtrip(self):
        transforms = ["sqrt"]
        mean = [0.0]
        scale = [1.0]
        raw = [9.0]
        std, ok = obs_raw_to_std(raw, transforms, mean, scale)
        self.assertTrue(ok)
        back, ok2 = obs_std_to_raw(std, transforms, mean, scale)
        self.assertTrue(ok2)
        self.assertAlmostEqual(back[0], 9.0)

    def test_log1p_at_zero(self):
        transforms = ["log1p"]
        mean = [0.0]
        scale = [1.0]
        std, ok = obs_raw_to_std([0.0], transforms, mean, scale)
        self.assertTrue(ok)
        self.assertAlmostEqual(std[0], 0.0)

    def test_unknown_transform_fails(self):
        std, ok = obs_raw_to_std([1.0], ["exp"], [0.0], [1.0])
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
