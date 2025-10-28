#!/usr/bin/env python3
"""
Test script for SDT .jnnx package
"""

import onnxruntime as ort
import pickle
import json
import numpy as np

def test_sdt_package():
    """Test the SDT .jnnx package"""
    
    print("Testing SDT .jnnx package...")
    
    # Load model
    try:
        session = ort.InferenceSession("model.onnx")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Load scaling parameters
    try:
        with open("scalers.pkl", 'rb') as f:
            scalers = pickle.load(f)
        print("✅ Scalers loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load scalers: {e}")
        return False
    
    # Load metadata
    try:
        with open("metadata.json", 'r') as f:
            metadata = json.load(f)
        print("✅ Metadata loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return False
    
    # Test prediction
    try:
        def predict(dprime, criterion):
            inputs = np.array([[dprime, criterion]], dtype=np.float32)
            scaled_inputs = scalers['x_scaler'].transform(inputs)
            outputs = session.run(None, {"input": scaled_inputs})[0]
            return scalers['y_scaler'].inverse_transform(outputs)
        
        # Test with known values
        result = predict(1.0, 0.0)
        hit_rate, false_alarm_rate = result[0][0], result[0][1]
        print(f"✅ Prediction successful")
        print(f"   dprime=1.0, criterion=0.0")
        print(f"   hit_rate={hit_rate:.3f}, false_alarm_rate={false_alarm_rate:.3f}")
        
        # Test with different values
        result2 = predict(2.0, -0.5)
        hit_rate2, false_alarm_rate2 = result2[0][0], result2[0][1]
        print(f"   dprime=2.0, criterion=-0.5")
        print(f"   hit_rate={hit_rate2:.3f}, false_alarm_rate={false_alarm_rate2:.3f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_sdt_package()
