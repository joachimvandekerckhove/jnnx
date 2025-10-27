// Simple test module for JAGS
// Auto-generated from jnnx template

#include <module/Module.h>
#include <function/ScalarFunction.h>
#include <vector>
#include <string>
#include <iostream>

namespace jags {
namespace sdt_emulator {

// Simple test function class
class SDT_Function : public ScalarFunction
{
public:
    SDT_Function() : ScalarFunction("sdt", 2)
    {
    }
    
    bool checkParameterValue(std::vector<double const *> const &args) const
    {
        return true;
    }
    
    double evaluate(std::vector<double const *> const &args) const
    {
        // Simple test: return sum of arguments
        double result = 0.0;
        for (unsigned int i = 0; i < 2; ++i) {
            result += *args[i];
        }
        return result;
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

} // namespace jags::sdt_emulator

// Module factory function
jags::sdt_emulator::SDT_EMULATOR_Module _sdt_emulator_module;

extern "C" {
    void jags_module_load()
    {
        std::cout << "The SDT emulator module is (c)2025 Joachim Vandekerckhove." << std::endl;
    }
}
}
