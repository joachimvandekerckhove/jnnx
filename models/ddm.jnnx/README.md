# DDM Neural Network Emulator

This package contains a trained neural network emulator for Drift Diffusion Model (DDM) models.

## Model Description

The DDM model predicts response accuracy, mean response time, and response time variance based on three parameters:
- **boundary**: Decision boundary parameter (threshold for decision)
- **drift**: Drift rate parameter (speed of evidence accumulation)
- **ndt**: Non-decision time parameter (time for stimulus encoding and response execution)

## Files

- `model.onnx`: Trained neural network model
- `scalers.pkl`: Scaling parameters (Python pickle format)
- `scalers.txt`: Human-readable scaling parameters
- `metadata.json`: Model specifications and metadata

## Installation Requirements

- Python 3.8+
- ONNX Runtime (`pip install onnxruntime`)
- NumPy (`pip install numpy`)

## Usage Example

```python
import onnxruntime as ort
import pickle
import json
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Load scaling parameters
with open("scalers.pkl", 'rb') as f:
    scalers = pickle.load(f)

# Load metadata
with open("metadata.json", 'r') as f:
    metadata = json.load(f)

# Transform inputs to [0,1] range
def transform_inputs(inputs):
    x_scaler = scalers['x_scaler']
    return x_scaler.transform(inputs)

# Transform outputs back to real-world values
def transform_outputs(outputs):
    y_scaler = scalers['y_scaler']
    return y_scaler.inverse_transform(outputs)

# Make prediction
def predict(boundary, drift, ndt):
    inputs = np.array([[boundary, drift, ndt]], dtype=np.float32)
    scaled_inputs = transform_inputs(inputs)
    outputs = session.run(None, {"input": scaled_inputs})[0]
    return transform_outputs(outputs)

# Example usage
result = predict(2.0, 1.0, 0.3)
accuracy, mean_rt, var_rt = result[0][0], result[0][1], result[0][2]
print(f"Accuracy: {accuracy:.3f}")
print(f"Mean RT: {mean_rt:.3f}s")
print(f"RT Variance: {var_rt:.3f}s²")
```

## Parameter Ranges

- **boundary**: 0.5 to 5.0
- **drift**: -5.0 to 5.0
- **ndt**: 0.0 to 1.0

## Output Ranges

- **accuracy**: 0.0 to 1.0 (probability)
- **mean_rt**: 0.0 to 10.0 (seconds)
- **var_rt**: 0.0 to 100.0 (seconds²)

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure ONNX Runtime is installed
2. **Invalid input**: Ensure parameters are within the specified ranges
3. **Scaling errors**: Verify scalers.pkl is not corrupted

### Validation

Test the model with known parameter values:
```python
# Test with boundary=2.0, drift=1.0, ndt=0.3
result = predict(2.0, 1.0, 0.3)
accuracy, mean_rt, var_rt = result[0][0], result[0][1], result[0][2]
# Expected: accuracy ≈ 0.85, mean_rt ≈ 1.2s, var_rt ≈ 0.5s²
```
