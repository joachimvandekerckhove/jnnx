"""
JNNX - JAGS Neural Network eXtension

Main package for integrating ONNX models with JAGS.
"""

__version__ = "1.0.0"
__author__ = "Joachim Vandekerckhove"
__email__ = "joachim@uci.edu"

from .core import JNNXPackage, JAGSModule, validate_jnnx_package, load_metadata, load_scalers

__all__ = [
    "JNNXPackage",
    "JAGSModule", 
    "validate_jnnx_package",
    "load_metadata",
    "load_scalers",
]
