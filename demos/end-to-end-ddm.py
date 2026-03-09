#!/usr/bin/env python3
"""
End-to-end DDM emulator demo: train → export (raw I/O) → .jnnx package → generate/compile → JAGS.

Runs the full pipeline:
1. Train a small PyTorch model (3 inputs → 3 outputs) in raw domain.
2. Export to ONNX with raw I/O (no scaling at graph boundaries).
3. Create a .jnnx package (metadata, model.onnx, scalers.json).
4. Validate package and generate JAGS module code.
5. Optionally compile (if ONNXRUNTIME_DIR is set) and run a short JAGS check.

Usage:
    python demos/end-to-end-ddm.py

Requirements: torch, onnx, numpy, jnnx. Optional: ONNXRUNTIME_DIR for compile, py2jags + JAGS for the JAGS step.
"""

import json
import os
import sys
from pathlib import Path

# Ensure repo root or installed jnnx is on path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

def main():
    print("=== 1. Train a minimal DDM emulator (raw I/O) ===\n")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Skipping train/export: torch not installed.")
        print("Using a pre-exported placeholder: create package from existing model if available.\n")
        # Could load models/sdt.jnnx or models/ddm.jnnx and copy to demo output
        _run_from_existing_package()
        return

    # Tiny MLP: 3 → 8 → 3, trained so that raw inputs produce raw outputs in expected ranges
    class TinyDDM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(3, 8)
            self.lin2 = nn.Linear(8, 3)

        def forward(self, x):
            x = torch.tanh(self.lin1(x))
            x = self.lin2(x)
            # Clamp outputs to plausible ranges (accuracy 0–1, mean_rt 0–10, var_rt 0–100)
            x = torch.cat([
                torch.sigmoid(x[:, :1]),
                torch.relu(x[:, 1:2]) + 0.3,
                torch.relu(x[:, 2:3]) + 0.01
            ], dim=1)
            return x

    model = TinyDDM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    np.random.seed(42)
    torch.manual_seed(42)
    for _ in range(200):
        # Sample in raw domain: drift [-3,3], boundary [0.5,3], ndt [0.1,1]
        x = torch.tensor(
            np.random.uniform(low=[-3, 0.5, 0.1], high=[3, 3, 1.0], size=(64, 3)),
            dtype=torch.float32
        )
        y = model(x).detach()  # fake targets in range
        y = y + 0.01 * torch.randn_like(y)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    out_dir = SCRIPT_DIR / "end_to_end_out"
    out_dir.mkdir(exist_ok=True)
    pkg_dir = out_dir / "ddm_demo.jnnx"
    pkg_dir.mkdir(exist_ok=True)

    print("=== 2. Export ONNX (raw I/O) ===\n")
    model.eval()
    onnx_path = out_dir / "model.onnx"
    dummy = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
    )
    print(f"Exported {onnx_path}")

    # Input/output ranges (raw domain) for metadata and scalers
    input_params = [
        {"name": "drift", "min": -3.0, "max": 3.0},
        {"name": "boundary", "min": 0.5, "max": 3.0},
        {"name": "ndt", "min": 0.1, "max": 1.0},
    ]
    output_params = [
        {"name": "accuracy", "min": 0.0, "max": 1.0},
        {"name": "mean_rt", "min": 0.0, "max": 10.0},
        {"name": "var_rt", "min": 0.0, "max": 100.0},
    ]

    print("=== 3. Create .jnnx package ===\n")
    import shutil
    shutil.copy2(onnx_path, pkg_dir / "model.onnx")
    metadata = {
        "model_name": "ddm_demo_emulator",
        "module_name": "ddm_demo_emulator",
        "function_name": "ddm_demo_emulator",
        "version": "1.0.0",
        "input_parameters": input_params,
        "output_parameters": output_params,
        "transformations": {
            "input_transform": "identity",
            "output_transforms": ["identity", "identity", "identity"],
        },
    }
    with open(pkg_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    scalers_json = {
        "version": "1.0",
        "input_scaler": {
            "type": "identity",
            "data_min": [p["min"] for p in input_params],
            "data_max": [p["max"] for p in input_params],
        },
        "output_scaler": {
            "type": "identity",
            "data_min": [p["min"] for p in output_params],
            "data_max": [p["max"] for p in output_params],
        },
        "metadata": {"created_by": "end-to-end-ddm.py", "note": "Raw I/O"},
    }
    with open(pkg_dir / "scalers.json", "w") as f:
        json.dump(scalers_json, f, indent=2)
    # Also write scalers.pkl for compatibility (same x_min/x_max/y_min/y_max)
    import pickle
    scalers_pkl = {
        "x_min": scalers_json["input_scaler"]["data_min"],
        "x_max": scalers_json["input_scaler"]["data_max"],
        "y_min": scalers_json["output_scaler"]["data_min"],
        "y_max": scalers_json["output_scaler"]["data_max"],
    }
    with open(pkg_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers_pkl, f)

    print(f"Package dir: {pkg_dir}")

    print("=== 4. Validate and generate JAGS module ===\n")
    from jnnx import JNNXPackage, JAGSModule

    pkg = JNNXPackage(str(pkg_dir))
    ok, errs = pkg.validate()
    if not ok:
        print("Validation failed:", errs)
        return 1
    print("Validation passed.")

    build_dir = out_dir / "build"
    mod = JAGSModule(pkg, str(build_dir))
    mod.generate_code()
    print(f"Generated code in {build_dir}")

    print("=== 5. Compile (optional) ===\n")
    if os.environ.get("ONNXRUNTIME_DIR"):
        ok, err = mod.compile()
        if ok:
            print("Compilation succeeded.")
        else:
            print("Compilation failed:", err)
    else:
        print("ONNXRUNTIME_DIR not set; skipping compile.")
        print("To compile: set ONNXRUNTIME_DIR and run make in", build_dir)

    print("=== 6. JAGS check (optional) ===\n")
    try:
        import py2jags
        fn = pkg.metadata["function_name"]
        mod_name = pkg.metadata["module_name"]
        # Only run JAGS if we compiled and installed
        so_file = build_dir / f"{mod_name}.so"
        if so_file.exists():
            print("JAGS/py2jags available and .so present; you can run JAGS with this module.")
        else:
            print("JAGS/py2jags available; install the module and use in JAGS as needed.")
    except ImportError:
        print("py2jags not installed; skip JAGS step.")

    print("\n=== Done ===\n")
    return 0


def _run_from_existing_package():
    """If torch is missing, validate and generate from an existing package if present."""
    from jnnx import JNNXPackage, JAGSModule
    for name in ["models/sdt.jnnx", "models/ddm.jnnx", "jnnx/models/ddm.jnnx"]:
        path = REPO_ROOT / name
        if path.exists():
            pkg = JNNXPackage(str(path))
            ok, errs = pkg.validate()
            print(f"Validated {path}: ok={ok}, errors={errs}")
            build_dir = SCRIPT_DIR / "end_to_end_out" / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            mod = JAGSModule(pkg, str(build_dir))
            mod.generate_code()
            print(f"Generated code in {build_dir}")
            return
    print("No existing .jnnx package found under models/ or jnnx/models/.")


if __name__ == "__main__":
    sys.exit(main() or 0)
