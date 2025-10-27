// JAGS module for ddm4_emulator
// Auto-generated from jnnx template

#include <module/Module.h>
#include <function/VectorFunction.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace jags {
namespace ddm4_emulator {

// Neural network function class
class DDM4_Function : public VectorFunction
{
private:
    Ort::Env env;
    Ort::Session* session;
    std::vector<float> x_min;
    std::vector<float> x_max;
    std::vector<float> y_min;
    std::vector<float> y_max;
    std::string scaling_path;
    bool scalers_loaded;
    
    bool load_scaling_parameters() {
        try {
            std::ifstream file(scaling_path);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot open scaling file: " + scaling_path);
            }
            
            // Read all values from the file
            std::vector<double> values;
            double value;
            while (file >> value) {
                values.push_back(value);
            }
            file.close();
            
            // Expected format: x_min, x_max, y_min, y_max
            int input_dim = 4;
            int output_dim = 5;
            int expected_values = 2 * input_dim + 2 * output_dim;
            
            if (values.size() != expected_values) {
                throw std::runtime_error("Expected " + std::to_string(expected_values) + 
                                       " scaling parameters, got " + std::to_string(values.size()));
            }
            
            // Assign values to member variables
            x_min.clear();
            x_max.clear();
            y_min.clear();
            y_max.clear();
            
            // x_min values (first input_dim values)
            for (int i = 0; i < input_dim; i++) {
                x_min.push_back(static_cast<float>(values[i]));
            }
            // x_max values (next input_dim values)
            for (int i = 0; i < input_dim; i++) {
                x_max.push_back(static_cast<float>(values[input_dim + i]));
            }
            // y_min values (next output_dim values)
            for (int i = 0; i < output_dim; i++) {
                y_min.push_back(static_cast<float>(values[2 * input_dim + i]));
            }
            // y_max values (last output_dim values)
            for (int i = 0; i < output_dim; i++) {
                y_max.push_back(static_cast<float>(values[2 * input_dim + output_dim + i]));
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load scaling parameters: " << e.what() << std::endl;
            return false;
        }
    }
    
public:
    DDM4_Function() : VectorFunction("ddm4", 4), scalers_loaded(false)
    {
        // Initialize ONNX Runtime
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ddm4_emulator");
        
        // Load ONNX model
        Ort::SessionOptions session_options;
        session = new Ort::Session(env, "/home/jovyan/project/models/ddm4.jnnx/ddm4_emulator.onnx", session_options);
        
        // Set scaling file path
        scaling_path = "/home/jovyan/project/models/ddm4.jnnx/build/ddm4_emulator_scaling.txt";
        
        // Load scaling parameters
        scalers_loaded = load_scaling_parameters();
        if (!scalers_loaded) {
            std::cerr << "Warning: Could not load scaling parameters, using default values" << std::endl;
            // Set default scaling (no scaling)
            x_min.resize(4, 0.0f);
            x_max.resize(4, 1.0f);
            y_min.resize(5, 0.0f);
            y_max.resize(5, 1.0f);
        }
    }
    
    ~DDM4_Function()
    {
        delete session;
    }
    
    bool checkParameterValue(std::vector<double const *> const &args,
                             std::vector<unsigned int> const &lengths) const
    {
        // Check for NaN or Inf values
        for (unsigned int i = 0; i < 4; ++i) {
            if (!std::isfinite(*args[i])) {
                return false;
            }
        }
        
        // Check input bounds
        std::vector<double> input_min = {0.5f, -5.0f, 0.0f, 0.0f};
        std::vector<double> input_max = {5.0f, 5.0f, 1.0f, 1.0f};
        
        for (unsigned int i = 0; i < 4; ++i) {
            double value = *args[i];
            // Skip bounds check for infinite limits
            if (std::isfinite(input_min[i]) && value < input_min[i]) {
                return false;
            }
            if (std::isfinite(input_max[i]) && value > input_max[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    void evaluate(double *value, 
                  std::vector<double const *> const &args,
                  std::vector<unsigned int> const &lengths) const
    {
        try {
            // Prepare input data
            std::vector<float> input_data(4);
            for (unsigned int i = 0; i < 4; ++i) {
                input_data[i] = static_cast<float>(*args[i]);
            }
            
            // Scale input to [0,1] range using MinMax scaling
            std::vector<float> input_scaled(4);
            for (unsigned int i = 0; i < 4; ++i) {
                if (x_max[i] - x_min[i] == 0.0f) {
                    input_scaled[i] = 0.5f;  // Default to middle of range
                } else {
                    input_scaled[i] = (input_data[i] - x_min[i]) / (x_max[i] - x_min[i]);
                }
            }
            
            // Create input tensor for ONNX Runtime
            std::vector<int64_t> input_shape = {1, 4};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_scaled.data(), input_scaled.size(),
                input_shape.data(), input_shape.size()
            );
            
            // Run neural network inference
            const char* input_names[] = {"input"};
            const char* output_names[] = {"output"};
            
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1
            );
            
            // Get output data from neural network
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            // Scale output back to real-world values (denormalize) and store in value array
            for (unsigned int i = 0; i < 5; ++i) {
                value[i] = static_cast<double>(output_data[i] * (y_max[i] - y_min[i]) + y_min[i]);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in neural network evaluation: " << e.what() << std::endl;
            // Return default values on error
            for (unsigned int i = 0; i < 5; ++i) {
                value[i] = 0.5;
            }
        }
    }
    
    unsigned int length(std::vector<unsigned int> const &arglengths,
                        std::vector<double const *> const &argvalues) const
    {
        // Return the number of output dimensions
        return 5;
    }
};

// Module class
class DDM4_EMULATOR_Module : public Module
{
public:
    DDM4_EMULATOR_Module() : Module("ddm4_emulator")
    {
        insert(new DDM4_Function());
    }
    
    ~DDM4_EMULATOR_Module()
    {
        std::vector<Function*> const &fvec = functions();
        for (unsigned int i = 0; i < fvec.size(); ++i) {
            delete fvec[i];
        }
    }
};

}} // namespace jags::ddm4_emulator

// Module factory function
jags::ddm4_emulator::DDM4_EMULATOR_Module _ddm4_emulator_module;

extern "C" {
    void jags_module_load()
    {
        std::cout << "The DDM4 emulator is being loaded. (c) 2025 Joachim Vandekerckhove" << std::endl;
    }
}

