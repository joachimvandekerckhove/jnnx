#!/usr/bin/env python3
"""
generate-module: Generate JAGS module code from templates and .jnnx metadata.

Usage: ./generate-module models/sdt.jnnx/
"""

import json
import os
import shutil
import sys
from pathlib import Path

from jnnx.capabilities import build_sl_config, get_capabilities, has_capability


def find_files(jnnx_dir):
    """Find required files in .jnnx directory."""
    jnnx_path = Path(jnnx_dir)
    if not jnnx_path.exists():
        print(f"Error: Directory {jnnx_dir} does not exist")
        sys.exit(1)

    if not jnnx_path.name.endswith('.jnnx'):
        print(f"Error: Directory {jnnx_dir} does not end with .jnnx")
        sys.exit(1)

    metadata_file = jnnx_path / "metadata.json"
    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {jnnx_dir}")
        sys.exit(1)

    onnx_file = jnnx_path / "model.onnx"
    if not onnx_file.exists():
        print(f"Error: model.onnx not found in {jnnx_dir}")
        sys.exit(1)

    pkl = jnnx_path / "scalers.pkl"
    js = jnnx_path / "scalers.json"
    if not pkl.exists() and not js.exists():
        print(f"Error: neither scalers.pkl nor scalers.json found in {jnnx_dir}")
        sys.exit(1)

    return metadata_file, onnx_file, jnnx_path


def load_metadata(metadata_file):
    """Load metadata.json configuration."""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {metadata_file}: {e}")
        sys.exit(1)


def extract_dimensions_from_metadata(metadata):
    """Extract input/output dimensions from metadata."""
    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])
    return len(input_params), len(output_params)


def extract_limits_from_metadata(metadata):
    """Extract input/output limits from metadata."""
    input_params = metadata.get('input_parameters', [])
    output_params = metadata.get('output_parameters', [])

    input_min = [param.get('min', 0.0) for param in input_params]
    input_max = [param.get('max', 1.0) for param in input_params]
    output_min = [param.get('min', 0.0) for param in output_params]
    output_max = [param.get('max', 1.0) for param in output_params]

    return input_min, input_max, output_min, output_max


def format_array(arr, use_double=False):
    """Format array for C++ code generation."""
    if not arr:
        return "{}"

    suffix = "" if use_double else "f"
    formatted = []
    for val in arr:
        if isinstance(val, str):
            if val in ("Inf", "inf"):
                formatted.append("1e38" + suffix)
            elif val in ("-Inf", "-inf"):
                formatted.append("-1e38" + suffix)
            else:
                try:
                    float_val = float(val)
                    formatted.append(f"{float_val:.10g}{suffix}")
                except ValueError:
                    formatted.append(f"{val}{suffix}")
        else:
            if val == float('inf'):
                formatted.append("1e38" + suffix)
            elif val == float('-inf'):
                formatted.append("-1e38" + suffix)
            else:
                formatted.append(f"{val:.17g}{suffix}")

    return "{" + ", ".join(formatted) + "}"


def format_int_array(arr):
    """Format integer list for C++ code generation."""
    if not arr:
        return "{}"
    return "{" + ", ".join(str(int(v)) for v in arr) + "}"


def _templates_dir():
    """Directory containing C++/Makefile templates."""
    return Path(__file__).resolve().parent.parent / "templates"


def _cpp_dir():
    return Path(__file__).resolve().parent.parent / "cpp"


def _build_module_registrations(sl_cfg):
    lines = []
    if sl_cfg["debug_exports"].get("predict", True):
        lines.append(
            f'insert(new PredictFunction(engine_, "{sl_cfg["predict_name"]}"));'
        )
    lines.append(
        f'insert(new PredictFunction(engine_, "{sl_cfg["emulator_name"]}"));'
    )
    if sl_cfg["debug_exports"].get("mean", True):
        lines.append(f'insert(new MeanFunction(engine_, "{sl_cfg["mean_name"]}"));')
    if sl_cfg["debug_exports"].get("omega1", True):
        lines.append(
            f'insert(new Omega1Function(engine_, "{sl_cfg["omega1_name"]}"));'
        )
    if sl_cfg["debug_exports"].get("omega_total", True):
        lines.append(
            'insert(new OmegaTotalFunction(engine_, '
            f'"{sl_cfg["omega_total_name"]}"));'
        )
    lines.append(
        f'insert(new LogdensFunction(engine_, "{sl_cfg["logdens_name"]}"));'
    )
    lines.append(
        f'insert(new SL_Distribution(engine_, "{sl_cfg["distribution_name"]}"));'
    )
    return "\n        ".join(lines)


def _write_build_manifest(output_dir, metadata, sl_cfg=None):
    manifest = {
        "module_name": metadata.get("module_name"),
        "capabilities": get_capabilities(metadata),
    }
    if sl_cfg:
        manifest["onnx_sha256"] = sl_cfg["onnx_sha256"]
        manifest["likelihood_sha256"] = sl_cfg["likelihood_sha256"]
        manifest["sigma_emu_baked"] = sl_cfg["sigma_emu_flat"]
        manifest["sl_p"] = sl_cfg["p"]
        manifest["obs_transform_sha256"] = sl_cfg["obs_transform_sha256"]
        manifest["obs_transform_baked"] = sl_cfg["obs_transform_baked"]
    (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))


def generate_module_code(metadata, onnx_file, output_dir, package_dir):
    """Generate C++ module code from templates."""
    sl_mode = has_capability(metadata, "synthetic_likelihood")
    template_name = "sl_module.cc.template" if sl_mode else "module.cc.template"
    template_file = _templates_dir() / template_name
    if not template_file.exists():
        print(f"Error: Template file not found: {template_file}")
        sys.exit(1)

    module_name = metadata.get('module_name')
    function_name = metadata.get('function_name')
    if not module_name or not function_name:
        print('Error: metadata.json must include module_name and function_name')
        sys.exit(1)

    function_class = f"{module_name.replace('_','').upper()}_Function"
    module_class = f"{module_name.replace('_','').upper()}_Module"

    input_dim, output_dim = extract_dimensions_from_metadata(metadata)
    input_min, input_max, output_min, output_max = extract_limits_from_metadata(metadata)

    model_name = metadata.get('model_name', module_name)
    banner = f"The {model_name} is being loaded. (c) 2025 Joachim Vandekerckhove"

    onnx_copy = output_dir / "model.onnx"
    shutil.copy2(onnx_file, onnx_copy)
    print(f"Copied ONNX model to: {onnx_copy}")

    replacements = {
        '{{MODULE_NAME}}': module_name,
        '{{FUNCTION_NAME}}': function_name,
        '{{FUNCTION_CLASS}}': function_class,
        '{{MODULE_CLASS}}': module_class,
        '{{INPUT_DIM}}': str(input_dim),
        '{{OUTPUT_DIM}}': str(output_dim),
        '{{ONNX_PATH}}': str(onnx_copy.absolute()),
        '{{INPUT_MIN}}': format_array(input_min, use_double=sl_mode),
        '{{INPUT_MAX}}': format_array(input_max, use_double=sl_mode),
        '{{OUTPUT_MIN}}': format_array(output_min),
        '{{OUTPUT_MAX}}': format_array(output_max),
        '{{BANNER_STRING}}': banner,
    }

    sl_cfg = None
    if sl_mode:
        sl_cfg = build_sl_config(metadata, package_dir)
        shutil.copy2(_cpp_dir() / "sl_math.h", output_dir / "sl_math.h")
        shutil.copy2(_cpp_dir() / "sl_math.cpp", output_dir / "sl_math.cc")
        replacements.update(
            {
                '{{P}}': str(sl_cfg["p"]),
                '{{N_CHOL}}': str(sl_cfg["n_chol"]),
                '{{SIGMA_EMU_FLAT}}': format_array(
                    sl_cfg["sigma_emu_flat"], use_double=True
                ),
                '{{COL_TRANSFORM}}': format_int_array(sl_cfg["col_transform_codes"]),
                '{{OBS_SCALER_MEAN}}': format_array(
                    sl_cfg["obs_scaler_mean"], use_double=True
                ),
                '{{OBS_SCALER_SCALE}}': format_array(
                    sl_cfg["obs_scaler_scale"], use_double=True
                ),
                '{{MODULE_REGISTRATIONS}}': _build_module_registrations(sl_cfg),
            }
        )
        print("Synthetic likelihood capability module")

    generated_content = template_file.read_text()
    for placeholder, value in replacements.items():
        generated_content = generated_content.replace(placeholder, value)

    output_file = output_dir / f"{module_name}.cc"
    output_file.write_text(generated_content)
    print(f"Generated: {output_file}")

    _write_build_manifest(output_dir, metadata, sl_cfg)
    return output_file


def generate_makefile(metadata, output_dir):
    """Generate Makefile from template."""
    template_file = _templates_dir() / "Makefile.template"
    if not template_file.exists():
        print(f"Error: Makefile template not found: {template_file}")
        sys.exit(1)

    module_name = metadata.get('module_name')
    if not module_name:
        print('Error: metadata.json must include module_name')
        sys.exit(1)

    install_dir = "/usr/lib/x86_64-linux-gnu/JAGS/modules-4/"
    onnx_default = os.environ.get('ONNXRUNTIME_DIR', '')
    sl_mode = has_capability(metadata, "synthetic_likelihood")

    if sl_mode:
        sources = f"{module_name}.cc sl_math.cc"
        sl_include = "-I."
        extra_cxxflags = ""
    else:
        sources = f"{module_name}.cc"
        sl_include = ""
        extra_cxxflags = ""

    replacements = {
        '{{MODULE_NAME}}': module_name,
        '{{INSTALL_DIR}}': install_dir,
        '{{ONNXRUNTIME_DIR_DEFAULT}}': onnx_default,
        '{{SOURCES}}': sources,
        '{{SL_INCLUDE}}': sl_include,
        '{{EXTRA_CXXFLAGS}}': extra_cxxflags,
    }

    generated_content = template_file.read_text()
    for placeholder, value in replacements.items():
        generated_content = generated_content.replace(placeholder, value)

    output_file = output_dir / "Makefile"
    output_file.write_text(generated_content)
    print(f"Generated: {output_file}")
    return output_file


def ensure_onnxruntime_in_tmp():
    """Ensure ONNX Runtime is available in tmp directory."""
    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)

    onnx_dir = tmp_dir / 'onnxruntime-linux-x64-1.23.2'
    if not onnx_dir.exists():
        print("ONNX Runtime not found in tmp/, extracting...")
        tgz_file = tmp_dir / 'onnxruntime-linux-x64-1.23.2.tgz'
        if tgz_file.exists():
            import tarfile
            with tarfile.open(tgz_file, 'r:gz') as tar:
                tar.extractall(tmp_dir)
            print(f"ONNX Runtime extracted to {onnx_dir}")
        else:
            print("Warning: ONNX Runtime archive not found in tmp/")
    else:
        print(f"ONNX Runtime found in {onnx_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: ./generate-module <jnnx-directory>")
        print("Example: ./generate-module models/sdt.jnnx/")
        sys.exit(1)

    jnnx_dir = sys.argv[1]

    metadata_file, onnx_file, package_dir = find_files(jnnx_dir)
    print("Found files:")
    print(f"  Metadata: {metadata_file}")
    print(f"  ONNX: {onnx_file}")
    print()

    metadata = load_metadata(metadata_file)
    print(f"Metadata loaded: {metadata.get('model_name', 'unnamed')}")
    print()

    ensure_onnxruntime_in_tmp()
    print()

    tmp_dir = Path('tmp')
    tmp_dir.mkdir(exist_ok=True)
    output_dir = tmp_dir / f"{Path(jnnx_dir).name}_build"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    print("Generating module code...")
    generate_module_code(metadata, onnx_file, output_dir, package_dir)
    print()

    print("Generating Makefile...")
    generate_makefile(metadata, output_dir)
    print()

    print("=" * 60)
    print("Module generation complete!")
    print()
    print("To compile and install:")
    print(f"  cd {output_dir}")
    print("  make")
    print("  sudo make install")


if __name__ == "__main__":
    main()
