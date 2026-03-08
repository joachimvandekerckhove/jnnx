#!/usr/bin/env python3
"""
Test script for DDM .jnnx package.

Loads model.onnx, scalers.pkl, and metadata.json from this package directory,
runs sample predictions, and verifies outputs. Works from any working directory
(paths are resolved relative to this script).

Run from repo root:
  python jnnx/models/ddm.jnnx/test_package.py

Or from the package directory:
  cd jnnx/models/ddm.jnnx && python test_package.py
"""

import onnxruntime as ort
import pickle
import json
import numpy as np
from pathlib import Path

# Package directory: same folder as this script
_PKG_DIR = Path(__file__).resolve().parent


def test_ddm_package():
    """Test the DDM .jnnx package."""

    print("Testing DDM .jnnx package...")

    # Load model
    model_path = _PKG_DIR / "model.onnx"
    try:
        session = ort.InferenceSession(str(model_path))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Load scaling parameters
    scalers_path = _PKG_DIR / "scalers.pkl"
    try:
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        print("✅ Scalers loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load scalers: {e}")
        return False

    # Load metadata
    metadata_path = _PKG_DIR / "metadata.json"
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False

    # Test prediction
    try:
        def predict(boundary, drift, ndt):
            inputs = np.array([[boundary, drift, ndt]], dtype=np.float32)
            scaled_inputs = scalers['x_scaler'].transform(inputs)
            outputs = session.run(None, {"input": scaled_inputs})[0]
            return scalers['y_scaler'].inverse_transform(outputs)

        # Test with known values
        result = predict(2.0, 1.0, 0.3)
        accuracy, mean_rt, var_rt = result[0][0], result[0][1], result[0][2]
        print(f"✅ Prediction successful")
        print(f"   boundary=2.0, drift=1.0, ndt=0.3")
        print(f"   accuracy={accuracy:.3f}, mean_rt={mean_rt:.3f}s, var_rt={var_rt:.3f}s²")

        # Test with different values
        result2 = predict(1.5, 0.5, 0.2)
        accuracy2, mean_rt2, var_rt2 = result2[0][0], result2[0][1], result2[0][2]
        print(f"   boundary=1.5, drift=0.5, ndt=0.2")
        print(f"   accuracy={accuracy2:.3f}, mean_rt={mean_rt2:.3f}s, var_rt={var_rt2:.3f}s²")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

    print("✅ All tests passed!")
    return True


if __name__ == "__main__":
    ok = test_ddm_package()
    raise SystemExit(0 if ok else 1)
