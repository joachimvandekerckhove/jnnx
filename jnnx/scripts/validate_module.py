#!/usr/bin/env python3
"""
validate-module: Test suite to validate a compiled JAGS module.

Usage: ./validate-module models/sdt.jnnx/ [--build-dir tmp/<pkg>_build]

Emulator tests (all packages):
1) Module loads in JAGS without error
2) Valid input vector works for mid-range values
3) Valid output vector (finite, correct dimension)
4) Invalid input size triggers error
5) Invalid input bounds triggers error
6) Numerical consistency with Python ONNX evaluation

Synthetic-likelihood tests (when capability declared):
SL 8.1-8.10, 8.8-8.9 (requires compiled + installed module)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import py2jags
    import onnxruntime as ort
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install py2jags onnxruntime")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jnnx.capabilities import get_capabilities, has_capability
from jnnx.core import JNNXPackage
from jnnx.sl_validation import run_sl_validation


def find_files(jnnx_dir: Path) -> Tuple[Path, Path, Path]:
    if not jnnx_dir.exists():
        print(f"Error: Directory {jnnx_dir} does not exist")
        sys.exit(1)
    if not jnnx_dir.name.endswith(".jnnx"):
        print(f"Error: Directory {jnnx_dir} does not end with .jnnx")
        sys.exit(1)

    metadata_file = jnnx_dir / "metadata.json"
    onnx_file = jnnx_dir / "model.onnx"
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {jnnx_dir}")
        sys.exit(1)
    if not onnx_file.exists():
        print(f"Error: model.onnx not found in {jnnx_dir}")
        sys.exit(1)

    scalers_file = jnnx_dir / "scalers.pkl"
    if not scalers_file.exists():
        scalers_file = jnnx_dir / "scalers.json"
        if not scalers_file.exists():
            print(f"Error: neither scalers.pkl nor scalers.json found in {jnnx_dir}")
            sys.exit(1)

    return metadata_file, onnx_file, scalers_file


def load_metadata(metadata_file: Path) -> Dict[str, Any]:
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {metadata_file}: {e}")
        sys.exit(1)


def load_scalers(scalers_file: Path) -> Dict[str, Any]:
    try:
        if scalers_file.suffix == ".json":
            with open(scalers_file, "r") as f:
                data = json.load(f)
            input_scaler = data.get("input_scaler", {})
            output_scaler = data.get("output_scaler", {})
            return {
                "x_min": input_scaler.get("data_min", []),
                "x_max": input_scaler.get("data_max", []),
                "y_min": output_scaler.get("data_min", []),
                "y_max": output_scaler.get("data_max", []),
            }

        with open(scalers_file, "rb") as f:
            scalers_data = pickle.load(f)

        if (
            isinstance(scalers_data, dict)
            and "x_scaler" in scalers_data
            and "y_scaler" in scalers_data
        ):
            x_scaler = scalers_data["x_scaler"]
            y_scaler = scalers_data["y_scaler"]
            return {
                "x_min": x_scaler.data_min_.tolist(),
                "x_max": x_scaler.data_max_.tolist(),
                "y_min": y_scaler.data_min_.tolist(),
                "y_max": y_scaler.data_max_.tolist(),
            }
        if isinstance(scalers_data, dict) and "x_min" in scalers_data:
            return scalers_data
        raise ValueError("Unknown scaler format")
    except Exception as e:
        print(f"Error: Could not load scalers from {scalers_file}: {e}")
        sys.exit(1)


def _mid_theta(metadata: Dict[str, Any]) -> List[float]:
    return [
        (p.get("min", 0.0) + p.get("max", 1.0)) / 2.0
        for p in metadata.get("input_parameters", [])
    ]


def _uses_raw_io(metadata: Dict[str, Any]) -> bool:
    transforms = metadata.get("transformations") or {}
    return transforms.get("input_transform", "minmax") == "identity"


def _run_jags(
    module_name: str,
    model_code: str,
    monitor: Optional[List[str]] = None,
) -> Any:
    kwargs = dict(
        model_string=model_code,
        data_dict={"n": 1},
        nchains=1,
        nsamples=1,
        nadapt=0,
        nburnin=0,
        modules=[module_name],
    )
    if monitor is not None:
        kwargs["monitorparams"] = monitor
    return py2jags.run_jags(**kwargs)


def _emulator_model(
    function_name: str,
    theta_str: str,
    output_dim: int,
    vector: bool = True,
) -> str:
    if vector and output_dim > 1:
        assign = f"result[1:{output_dim}] <- {function_name}({theta_str})"
    else:
        assign = f"result <- {function_name}({theta_str})"
    return f"""
    model {{
        {assign}
        dummy ~ dnorm(0, 1)
    }}
    """


def _onnx_forward(
    session: ort.InferenceSession,
    metadata: Dict[str, Any],
    scalers: Dict[str, Any],
    test_inputs: List[float],
) -> np.ndarray:
    """Match JAGS module contract: raw theta in, ONNX graph handles scaling."""
    input_tensor = np.array([test_inputs], dtype=np.float32)
    return session.run(None, {"input": input_tensor})[0][0]


def test_module_loading(metadata: Dict[str, Any]) -> bool:
    print("Test 1: Module loading...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        output_dim = len(metadata.get("output_parameters", []))
        theta = _mid_theta(metadata)
        theta_str = ", ".join(str(x) for x in theta)
        model_code = _emulator_model(function_name, theta_str, output_dim)
        _run_jags(module_name, model_code, monitor=["result"])
        print("  ✓ Module loaded successfully in JAGS")
        return True
    except Exception as e:
        print(f"  ✗ Error loading module: {e}")
        return False


def test_valid_input_range(metadata: Dict[str, Any]) -> bool:
    print("Test 2: Valid input range...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        output_dim = len(metadata.get("output_parameters", []))
        test_inputs = _mid_theta(metadata)
        theta_str = ", ".join(str(x) for x in test_inputs)
        model_code = _emulator_model(function_name, theta_str, output_dim)
        _run_jags(module_name, model_code, monitor=["result"])
        print(f"  ✓ Valid input range works: {test_inputs}")
        return True
    except Exception as e:
        print(f"  ✗ Error with valid input range: {e}")
        return False


def test_valid_output(metadata: Dict[str, Any]) -> bool:
    print("Test 3: Valid output...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        output_dim = len(metadata.get("output_parameters", []))
        test_inputs = _mid_theta(metadata)
        theta_str = ", ".join(str(x) for x in test_inputs)
        model_code = _emulator_model(function_name, theta_str, output_dim)
        chains = _run_jags(module_name, model_code, monitor=["result"])

        results = []
        for i in range(output_dim):
            param_name = f"result_{i+1}"
            if param_name in chains.parameter_names:
                results.append(chains.get_samples(param_name)[0])
            elif output_dim == 1 and "result" in chains.parameter_names:
                results.append(chains.get_samples("result")[0])

        if len(results) != output_dim:
            print(f"  ✗ Output dimension mismatch: expected {output_dim}, got {len(results)}")
            return False
        if np.any(np.isnan(results)) or np.any(np.isinf(results)):
            print(f"  ✗ Invalid output values (NaN or Inf): {results}")
            return False
        print(f"  ✓ Valid output: {results}")
        return True
    except Exception as e:
        print(f"  ✗ Error with valid output: {e}")
        return False


def test_invalid_input_size(metadata: Dict[str, Any]) -> bool:
    print("Test 4: Invalid input size...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        input_dim = len(metadata.get("input_parameters", []))
        wrong_args = ", ".join(["1.0"] * (input_dim + 2))
        output_dim = len(metadata.get("output_parameters", []))
        model_code = _emulator_model(function_name, wrong_args, output_dim)
        _run_jags(module_name, model_code, monitor=["result"])
        print("  ✗ Invalid input size did not trigger error (unexpected)")
        return False
    except Exception as e:
        print(f"  ✓ Invalid input size correctly triggered error: {e}")
        return True


def test_invalid_input_bounds(metadata: Dict[str, Any]) -> bool:
    print("Test 5: Invalid input bounds...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        output_dim = len(metadata.get("output_parameters", []))
        test_inputs = [
            p.get("max", 1.0) + 100.0 for p in metadata.get("input_parameters", [])
        ]
        theta_str = ", ".join(str(x) for x in test_inputs)
        model_code = _emulator_model(function_name, theta_str, output_dim)
        _run_jags(module_name, model_code, monitor=["result"])
        print("  ✗ Invalid input bounds did not trigger error (unexpected)")
        return False
    except Exception as e:
        print(f"  ✓ Invalid input bounds correctly triggered error: {e}")
        return True


def test_numerical_consistency(
    metadata: Dict[str, Any],
    onnx_file: Path,
    scalers: Dict[str, Any],
) -> bool:
    print("Test 6: Numerical consistency...")
    try:
        module_name = metadata["module_name"]
        function_name = metadata["function_name"]
        output_dim = len(metadata.get("output_parameters", []))
        test_inputs = _mid_theta(metadata)
        theta_str = ", ".join(str(x) for x in test_inputs)

        session = ort.InferenceSession(str(onnx_file))
        python_output = _onnx_forward(session, metadata, scalers, test_inputs)

        model_code = _emulator_model(function_name, theta_str, output_dim)
        chains = _run_jags(module_name, model_code, monitor=["result"])

        jags_output = []
        for i in range(output_dim):
            param_name = f"result_{i+1}"
            if param_name in chains.parameter_names:
                jags_output.append(chains.get_samples(param_name)[0])

        if len(jags_output) != output_dim:
            print(f"  ✗ JAGS output dimension mismatch: expected {output_dim}, got {len(jags_output)}")
            return False

        max_diff = max(abs(a - b) for a, b in zip(python_output, jags_output))
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"  ✓ Numerical consistency: max difference = {max_diff:.2e}")
            return True
        print(f"  ✗ Numerical inconsistency: max difference = {max_diff:.2e}")
        print(f"    Python: {python_output}")
        print(f"    JAGS:   {jags_output}")
        return False
    except Exception as e:
        print(f"  ✗ Error in numerical consistency test: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jnnx_dir", type=Path, help="Path to .jnnx package")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Generated build directory (default tmp/<pkg>_build)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to SL regression fixture JSON (overrides search paths)",
    )
    args = parser.parse_args()

    jnnx_dir = args.jnnx_dir.resolve()
    metadata_file, onnx_file, scalers_file = find_files(jnnx_dir)
    print(f"Validating module for: {jnnx_dir}")
    print(f"  Metadata: {metadata_file}")
    print(f"  ONNX: {onnx_file}")
    print(f"  Scalers: {scalers_file}")
    print()

    metadata = load_metadata(metadata_file)
    scalers = load_scalers(scalers_file)
    capabilities = get_capabilities(metadata)

    print(f"Model: {metadata.get('model_name', 'unnamed')}")
    print(f"Module: {metadata.get('module_name', 'unset')}  Function: {metadata.get('function_name', 'unset')}")
    print(f"Capabilities: {capabilities}")
    print(f"Version: {metadata.get('version', 'unknown')}")
    print()

    emulator_tests = [
        test_module_loading(metadata),
        test_valid_input_range(metadata),
        test_valid_output(metadata),
        test_invalid_input_size(metadata),
        test_invalid_input_bounds(metadata),
        test_numerical_consistency(metadata, onnx_file, scalers),
    ]
    emulator_passed = sum(emulator_tests)
    emulator_total = len(emulator_tests)

    sl_passed = 0
    sl_total = 0
    if has_capability(metadata, "synthetic_likelihood"):
        print("Synthetic likelihood validation:")
        package = JNNXPackage(str(jnnx_dir))
        build_dir = args.build_dir or (ROOT / "tmp" / f"{jnnx_dir.name}_build")
        sl_passed, sl_total = run_sl_validation(
            package, build_dir, fixture_path=args.fixture
        )
        print()

    total_passed = emulator_passed + sl_passed
    total_tests = emulator_total + sl_total

    print("=" * 50)
    print(f"Emulator: {emulator_passed}/{emulator_total} tests passed")
    if sl_total:
        print(f"Synthetic likelihood: {sl_passed}/{sl_total} tests passed")
    print(f"Validation Summary: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("✓ All tests passed! Module is working correctly.")
        sys.exit(0)
    print("✗ Some tests failed. Please check the module.")
    sys.exit(1)


if __name__ == "__main__":
    main()
