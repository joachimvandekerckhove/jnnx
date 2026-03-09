"""
JNNX - JAGS Neural Network eXtension

Main package for integrating ONNX models with JAGS.
"""

try:
    from importlib.metadata import version
    __version__ = version("jnnx")
except Exception:
    __version__ = "0.0.0+unknown"

__author__ = "Joachim Vandekerckhove"
__email__ = "joachim@uci.edu"

from .core import JNNXPackage, JAGSModule, validate_jnnx_package, load_metadata, load_scalers

__all__ = [
    "__version__",
    "JNNXPackage",
    "JAGSModule",
    "validate_jnnx_package",
    "load_metadata",
    "load_scalers",
]
