# SDT Neural Network Emulator

This package contains a trained neural network emulator for Signal Detection Theory (SDT) models.

## Model Description

The SDT model predicts hit rates and false alarm rates based on two parameters:
- **dprime**: Signal sensitivity (discriminability)
- **criterion**: Response bias (decision threshold)

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
def predict(dprime, criterion):
    inputs = np.array([[dprime, criterion]], dtype=np.float32)
    scaled_inputs = transform_inputs(inputs)
    outputs = session.run(None, {"input": scaled_inputs})[0]
    return transform_outputs(outputs)

# Example usage
result = predict(1.5, 0.2)
hit_rate, false_alarm_rate = result[0][0], result[0][1]
print(f"Hit rate: {hit_rate:.3f}")
print(f"False alarm rate: {false_alarm_rate:.3f}")
```

## Parameter Ranges

- **dprime**: -5.0 to 5.0
- **criterion**: -5.0 to 5.0

## Output Ranges

- **hit_rate**: 0.0 to 1.0 (probability)
- **false_alarm_rate**: 0.0 to 1.0 (probability)

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure ONNX Runtime is installed
2. **Invalid input**: Ensure parameters are within the specified ranges
3. **Scaling errors**: Verify scalers.pkl is not corrupted

### Validation

Test the model with known parameter values:
```python
# Test with dprime=1.0, criterion=0.0
result = predict(1.0, 0.0)
hit_rate, false_alarm_rate = result[0][0], result[0][1]
# Expected: hit_rate ≈ 0.84, false_alarm_rate ≈ 0.50
```
