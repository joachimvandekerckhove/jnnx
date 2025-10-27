// JAGS module for sdt_emulator
// Auto-generated from jnnx template

#include <Module.h>
#include <function/ScalarFunction.h>
#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <vector>
#include <string>

namespace jags {
namespace sdt_emulator {

// Neural network function class
class SDT_Function : public ScalarFunction
{
private:
    Ort::Env env;
    Ort::Session* session;
    std::vector<float> input_min;
    std::vector<float> input_max;
    std::vector<float> output_min;
    std::vector<float> output_max;
    
public:
    SDT_Function() : ScalarFunction("sdt", 2)
    {
        // Initialize ONNX Runtime
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "sdt_emulator");
        
        // Load ONNX model
        Ort::SessionOptions session_options;
        session = new Ort::Session(env, "/home/jovyan/project/models/sdt.jnnx/sdt_emulator.onnx", session_options);
        
        // Load scalers from PyTorch file
        // TODO: Implement scaler loading
        
        // Set bounds
        input_min = {-1e+308, -1e+308};
        input_max = {1e+308, 1e+308};
        output_min = {0, 0};
        output_max = {1, 1};
    }
    
    ~SDT_Function()
    {
        delete session;
    }
    
    bool checkParameterValue(std::vector<double const *> const &args) const
    {
        // Check input bounds
        for (unsigned int i = 0; i < 2; ++i) {
            double val = *args[i];
            if (val < input_min[i] || val > input_max[i]) {
                return false;
            }
        }
        return true;
    }
    
    double evaluate(std::vector<double const *> const &args) const
    {
        // Prepare input tensor
        std::vector<float> input_data(2);
        for (unsigned int i = 0; i < 2; ++i) {
            input_data[i] = static_cast<float>(*args[i]);
        }
        
        // Run inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 2};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );
        
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Extract output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        // Return first output (for scalar function)
        return static_cast<double>(output_data[0]);
    }
};

// Module class
class SDT_EMULATOR_Module : public Module
{
public:
    SDT_EMULATOR_Module() : Module("sdt_emulator")
    {
        insert(new SDT_Function());
    }
    
    ~SDT_EMULATOR_Module()
    {
        std::vector<Function*> const &fvec = functions();
        for (unsigned int i = 0; i < fvec.size(); ++i) {
            delete fvec[i];
        }
    }
};

}} // namespace jags::sdt_emulator

// Module factory function
jags::sdt_emulator::SDT_EMULATOR_Module _sdt_emulator_module;

extern "C" {
    void jags_module_load()
    {
        std::cout << "The SDT emulator module is (c)2025 Joachim Vandekerckhove." << std::endl;
    }
}

