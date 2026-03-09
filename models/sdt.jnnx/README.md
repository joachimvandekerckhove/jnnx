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
import numpy as np

# Load model (raw I/O: scaling is baked into the ONNX graph)
session = ort.InferenceSession("model.onnx")

# Pass raw-domain inputs directly; outputs are in original domain
def predict(d_prime, criterion):
    inputs = np.array([[d_prime, criterion]], dtype=np.float32)
    outputs = session.run(None, {"input": inputs})[0]
    return outputs

result = predict(1.5, 0.0)
hit_rate, fa_rate = result[0][0], result[0][1]
```

### C++ Loading
```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// ONNX model uses raw I/O; no external scaling needed.
class ModelLoader {
private:
    Ort::Env env;
    Ort::Session session;

public:
    ModelLoader(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "ModelLoader"),
          session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {}

    std::vector<float> predict(float d_prime, float criterion) {
        std::vector<float> inputs = {d_prime, criterion};
        // ... ONNX Runtime inference code (raw values in, raw values out) ...
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
- `model.onnx`: Trained neural network model (raw I/O; scaling baked into the graph)
- `scalers.pkl`: Scaling parameters (Python pickle)
- `metadata.json`: Model specifications and metadata
- `README.md`: This documentation
