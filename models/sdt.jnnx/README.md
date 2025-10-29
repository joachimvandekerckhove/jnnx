# SDT Cognitive Model Emulator

## Description
Signal Detection Theory model mapping sensitivity and criterion to hit/false alarm rates

## Installation Requirements
- ONNX Runtime: Version 1.23.0 or compatible
- Python: 3.8+ (if using Python interface)
- C++: C++17+ (if using C++ interface)

## Usage Examples

### Python Loading
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
    x_min = np.array(scalers['x_min'])
    x_max = np.array(scalers['x_max'])
    return (inputs - x_min) / (x_max - x_min)

# Transform outputs back to real-world values
def transform_outputs(outputs):
    y_min = np.array(scalers['y_min'])
    y_max = np.array(scalers['y_max'])
    return outputs * (y_max - y_min) + y_min

# Make prediction
def predict(boundary, drift, ndt):
    inputs = np.array([[boundary, drift, ndt]], dtype=np.float32)
    scaled_inputs = transform_inputs(inputs)
    outputs = session.run(None, {"input": scaled_inputs})[0]
    return transform_outputs(outputs)
```

### C++ Loading
```cpp
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>

class ModelLoader {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<float> x_min, x_max, y_min, y_max;
    
public:
    ModelLoader(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING, "ModelLoader"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
        load_scalers();
    }
    
    void load_scalers() {
        // Load scaling parameters from scalers.txt
        std::ifstream file("scalers.txt");
        std::vector<float> params;
        float value;
        while (file >> value) {
            params.push_back(value);
        }
        
        // Split into x_min, x_max, y_min, y_max
        size_t n_inputs = params.size() / 4;
        x_min.assign(params.begin(), params.begin() + n_inputs);
        x_max.assign(params.begin() + n_inputs, params.begin() + 2*n_inputs);
        y_min.assign(params.begin() + 2*n_inputs, params.begin() + 3*n_inputs);
        y_max.assign(params.begin() + 3*n_inputs, params.end());
    }
    
    std::vector<float> predict(float boundary, float drift, float ndt) {
        // Transform inputs
        std::vector<float> inputs = {boundary, drift, ndt};
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i] = (inputs[i] - x_min[i]) / (x_max[i] - x_min[i]);
        }
        
        // Run inference
        // ... ONNX Runtime inference code ...
        
        // Transform outputs
        // ... output transformation code ...
    }
};
```

## API Reference

### Input Parameters
- **d_prime**: -3.0 to 3.0
- **criterion**: -2.0 to 2.0

### Output Parameters
- **hit_rate**: Model output
- **fa_rate**: Model output

## Model Information
- Architecture: [64, 32]
- Training epochs: 98
- Best validation loss: 0.000045
- Trained: Unknown

## Troubleshooting

### Common Issues
1. **ONNX Runtime Error**: Ensure ONNX Runtime version 1.23.0+ is installed
2. **File Not Found**: Verify all required files are present in the package
3. **Input Validation**: Always validate inputs against parameter ranges
4. **Pickle Security**: Only load scalers.pkl from trusted sources

### Platform Support
- **Primary**: Linux x64
- **Secondary**: Windows x64, macOS x64 (with compatible ONNX Runtime)

## Files
- `model.onnx`: Trained neural network model
- `scalers.pkl`: Scaling parameters (Python pickle)
- `scalers.txt`: Human-readable scaling parameters
- `metadata.json`: Model specifications and metadata
- `README.md`: This documentation
