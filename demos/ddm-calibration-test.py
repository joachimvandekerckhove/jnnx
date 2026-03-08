#!/usr/bin/env python3
"""
DDM Calibration Test

This script performs a calibration test for the DDM (Drift Diffusion Model) JAGS module.
The calibration test verifies that Bayesian 95% credible intervals are well-calibrated,
i.e., they should contain the true parameter value approximately 95% of the time.

Overview:
1. Generate random true parameters
2. Simulate "observed" summary statistics using the DDM emulator
3. Run MCMC to estimate parameters and compute 95% credible intervals
4. Check if true values fall within their respective credible intervals
5. Repeat many times and report coverage percentage
"""

import py2jags
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List
from jnnx import JNNXPackage

# Setup and Configuration
print("=" * 60)
print("DDM Calibration Test")
print("=" * 60)

# Load module information
pkg = JNNXPackage('/home/jovyan/project/models/ddm.jnnx')
MODULE_NAME = pkg.metadata['module_name']
FUNCTION_NAME = pkg.metadata['function_name']
print(f'\nUsing module: {MODULE_NAME}, function: {FUNCTION_NAME}')

# Parameter ranges based on metadata
# drift: -3 to 3
# boundary: 0.5 to 3.0
# ndt: 0.1 to 1.0

# Number of iterations
N_ITERATIONS = 200
print(f"\nRunning {N_ITERATIONS} iterations for calibration test...")
print("Each iteration will check if true values fall within 95% credible intervals.\n")

@dataclass
class CalibrationResult:
    """Stores results from a single calibration test iteration."""
    true_drift: float
    true_boundary: float
    true_ndt: float
    drift_ci_lower: float
    drift_ci_upper: float
    boundary_ci_lower: float
    boundary_ci_upper: float
    ndt_ci_lower: float
    ndt_ci_upper: float
    drift_in_ci: bool
    boundary_in_ci: bool
    ndt_in_ci: bool
    iteration_time: float
    max_rhat: float
    converged: bool
    
    @property
    def all_in_ci(self) -> bool:
        """Check if all parameters are within their credible intervals."""
        return self.drift_in_ci and self.boundary_in_ci and self.ndt_in_ci

# Storage for results
results: List[CalibrationResult] = []

# JAGS model for parameter estimation
recovery_model_template = '''
model {
    # Priors on parameters
    drift ~ dnorm(0, 0.1)  # Prior centered at 0, relatively uninformative
    boundary ~ dgamma(2, 1)  # Prior for boundary (positive, mean ~2)
    ndt ~ dbeta(2, 5)  # Prior for NDT (0-1 range, mean ~0.29)
    
    # Scale ndt to appropriate range (0.1 to 1.0)
    ndt_scaled <- 0.1 + ndt * 0.9
    
    # Use emulator to predict statistics
    predicted_stats <- {FUNCTION_NAME}(drift, boundary, ndt_scaled)
    
    # Likelihood: compare predicted to observed statistics
    # Compute precision based on observed statistics and sample size N
    precision_mean_rt <- N / predicted_stats[3]
    precision_var_rt <- (N+1) / (2*predicted_stats[3]^2)

    # Using normal distributions with high precision for matching
    observed_accuracy_count ~ dbinom(predicted_stats[1], N)
    observed_mean_rt ~ dnorm(predicted_stats[2], precision_mean_rt)
    observed_var_rt ~ dnorm(predicted_stats[3], precision_var_rt)
}
'''

# Run iterations
np.random.seed(42)  # For reproducibility
start_total = time.time()

sample_size = 1000

for iteration in range(N_ITERATIONS):
    iter_start = time.time()
    
    # Step 1: Generate random true parameters
    true_drift = np.random.uniform(-3.0, 3.0)
    true_boundary = np.random.uniform(0.5, 3.0)
    true_ndt = np.random.uniform(0.1, 1.0)
    
    # Step 2: Simulate "observed" data
    model_string = f'''
model {{
    observed_stats <- {FUNCTION_NAME}({true_drift}, {true_boundary}, {true_ndt})
    dummy ~ dnorm(0, 1)
}}
'''
    
    result = py2jags.run_jags(
        model_string=model_string,
        data_dict={'n': 1},
        nchains=1, nsamples=1, nadapt=0, nburnin=0,
        monitorparams=['observed_stats'],
        modules=[MODULE_NAME],
        verbosity=0
    )
    
    # Extract observed statistics
    observed_accuracy_count = np.round(sample_size * result.get_samples('observed_stats_1')[0])
    observed_mean_rt = result.get_samples('observed_stats_2')[0]
    observed_var_rt = result.get_samples('observed_stats_3')[0]
    
    # Step 3: Estimate parameters and compute credible intervals
    recovery_model = recovery_model_template.replace('{FUNCTION_NAME}', FUNCTION_NAME)
    
    data = {
        'observed_accuracy_count': observed_accuracy_count,
        'observed_mean_rt': observed_mean_rt,
        'observed_var_rt': observed_var_rt,
        'N': sample_size
    }
    
    # Run MCMC
    chains = py2jags.run_jags(
        model_string=recovery_model,
        data_dict=data,
        nchains=4,
        nsamples=1000,
        nadapt=500,
        nburnin=500,
        monitorparams=['drift', 'boundary', 'ndt_scaled'],
        parallel=True,
        maxcores=4,
        modules=[MODULE_NAME],
        verbosity=0
    )
    
    # Step 4: Extract posterior samples and compute credible intervals
    drift_samples = chains.get_samples('drift')
    boundary_samples = chains.get_samples('boundary')
    ndt_samples = chains.get_samples('ndt_scaled')
    
    # Compute 95% credible intervals
    drift_ci_lower = np.percentile(drift_samples, 2.5)
    drift_ci_upper = np.percentile(drift_samples, 97.5)
    boundary_ci_lower = np.percentile(boundary_samples, 2.5)
    boundary_ci_upper = np.percentile(boundary_samples, 97.5)
    ndt_ci_lower = np.percentile(ndt_samples, 2.5)
    ndt_ci_upper = np.percentile(ndt_samples, 97.5)
    
    # Check if true values fall within credible intervals
    drift_in_ci = drift_ci_lower <= true_drift <= drift_ci_upper
    boundary_in_ci = boundary_ci_lower <= true_boundary <= boundary_ci_upper
    ndt_in_ci = ndt_ci_lower <= true_ndt <= ndt_ci_upper
    
    # Check convergence
    max_rhat = chains.max_rhat()
    converged = chains.converged()
    
    iter_time = time.time() - iter_start
    
    # Store results as an object
    result = CalibrationResult(
        true_drift=true_drift,
        true_boundary=true_boundary,
        true_ndt=true_ndt,
        drift_ci_lower=drift_ci_lower,
        drift_ci_upper=drift_ci_upper,
        boundary_ci_lower=boundary_ci_lower,
        boundary_ci_upper=boundary_ci_upper,
        ndt_ci_lower=ndt_ci_lower,
        ndt_ci_upper=ndt_ci_upper,
        drift_in_ci=drift_in_ci,
        boundary_in_ci=boundary_in_ci,
        ndt_in_ci=ndt_in_ci,
        iteration_time=iter_time,
        max_rhat=max_rhat,
        converged=converged
    )
    results.append(result)
    
    if max_rhat > 1.1:
        print(f"Convergence warning: Max Rhat = {max_rhat:.3f} > 1.1 for iteration {iteration + 1}")
    elif not converged:
        print(f"Convergence warning: Chains not converged (Rhat = {max_rhat:.3f}) for iteration {iteration + 1}")
    
    # Progress update every 10 iterations
    if (iteration + 1) % 10 == 0:
        avg_time = np.mean([r.iteration_time for r in results])
        elapsed = time.time() - start_total
        remaining = avg_time * (N_ITERATIONS - iteration - 1)
        print(f"Iteration {iteration + 1}/{N_ITERATIONS} | "
              f"Time: {iter_time:.2f}s | "
              f"Avg: {avg_time:.2f}s | "
              f"Elapsed: {elapsed:.1f}s | "
              f"Remaining: ~{remaining:.1f}s")

total_time = time.time() - start_total
print(f"\n{'='*60}")
print(f"Completed {N_ITERATIONS} iterations in {total_time:.2f} seconds")
avg_time = np.mean([r.iteration_time for r in results])
print(f"Average time per iteration: {avg_time:.2f} seconds")
print(f"{'='*60}")

# Analyze calibration results
print("\n" + "=" * 60)
print("Calibration Analysis")
print("=" * 60)

# Extract arrays from results objects
converged_arr = np.array([r.converged for r in results])
drift_in_ci_arr = np.array([r.drift_in_ci for r in results])
boundary_in_ci_arr = np.array([r.boundary_in_ci for r in results])
ndt_in_ci_arr = np.array([r.ndt_in_ci for r in results])

# Filter to only converged runs
converged_mask = converged_arr == True
n_converged = np.sum(converged_mask)
n_non_converged = N_ITERATIONS - n_converged

print(f"\nConvergence Summary:")
print(f"  Converged: {n_converged}/{N_ITERATIONS} ({100*n_converged/N_ITERATIONS:.1f}%)")
print(f"  Non-converged: {n_non_converged}/{N_ITERATIONS} ({100*n_non_converged/N_ITERATIONS:.1f}%)")
print(f"\nCalibration results below are calculated ONLY for converged runs.")

if n_converged > 0:
    # Calculate coverage for converged runs only
    drift_coverage = np.sum(drift_in_ci_arr[converged_mask]) / n_converged * 100
    boundary_coverage = np.sum(boundary_in_ci_arr[converged_mask]) / n_converged * 100
    ndt_coverage = np.sum(ndt_in_ci_arr[converged_mask]) / n_converged * 100
    
    print(f"\n95% Credible Interval Coverage (converged runs only, n={n_converged}):")
    print(f"  Drift:    {drift_coverage:.1f}% ({np.sum(drift_in_ci_arr[converged_mask])}/{n_converged})")
    print(f"  Boundary: {boundary_coverage:.1f}% ({np.sum(boundary_in_ci_arr[converged_mask])}/{n_converged})")
    print(f"  NDT:      {ndt_coverage:.1f}% ({np.sum(ndt_in_ci_arr[converged_mask])}/{n_converged})")
    print(f"\n  Overall:  {(drift_coverage + boundary_coverage + ndt_coverage) / 3:.1f}% average coverage")
    print(f"  Expected: 95.0% (ideal calibration)")
    
    # Calculate coverage for all runs (including non-converged)
    drift_coverage_all = np.sum(drift_in_ci_arr) / N_ITERATIONS * 100
    boundary_coverage_all = np.sum(boundary_in_ci_arr) / N_ITERATIONS * 100
    ndt_coverage_all = np.sum(ndt_in_ci_arr) / N_ITERATIONS * 100
    
    print(f"\n95% Credible Interval Coverage (all runs, n={N_ITERATIONS}):")
    print(f"  Drift:    {drift_coverage_all:.1f}% ({np.sum(drift_in_ci_arr)}/{N_ITERATIONS})")
    print(f"  Boundary: {boundary_coverage_all:.1f}% ({np.sum(boundary_in_ci_arr)}/{N_ITERATIONS})")
    print(f"  NDT:      {ndt_coverage_all:.1f}% ({np.sum(ndt_in_ci_arr)}/{N_ITERATIONS})")
else:
    print("\nWARNING: No converged runs! Cannot calculate calibration statistics.")

# Visualize calibration results
print("\n" + "=" * 60)
print("Visualizing Calibration Results")
print("=" * 60)

# Create calibration plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

param_data = [
    ([r.true_drift for r in results], [r.drift_ci_lower for r in results], [r.drift_ci_upper for r in results], 
     drift_in_ci_arr, converged_arr, 'Drift', 'drift'),
    ([r.true_boundary for r in results], [r.boundary_ci_lower for r in results], [r.boundary_ci_upper for r in results],
     boundary_in_ci_arr, converged_arr, 'Boundary', 'boundary'),
    ([r.true_ndt for r in results], [r.ndt_ci_lower for r in results], [r.ndt_ci_upper for r in results],
     ndt_in_ci_arr, converged_arr, 'NDT', 'ndt')
]

for idx, (true_vals, ci_lower, ci_upper, in_ci, converged, title, param_name) in enumerate(param_data):
    ax = axes[idx]
    
    true_vals = np.array(true_vals)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    in_ci = np.array(in_ci)
    converged = np.array(converged)
    
    # Separate converged and non-converged
    conv_mask = converged == True
    nonconv_mask = ~conv_mask
    
    # Plot converged runs
    if np.sum(conv_mask) > 0:
        true_conv = true_vals[conv_mask]
        in_ci_conv = in_ci[conv_mask]
        
        # Points where true value is in CI (green)
        in_ci_mask = in_ci_conv == True
        if np.sum(in_ci_mask) > 0:
            ax.scatter(true_conv[in_ci_mask], np.arange(len(true_conv))[in_ci_mask], 
                      alpha=0.6, s=30, color='green', marker='o', label='In CI (converged)')
        
        # Points where true value is NOT in CI (red)
        out_ci_mask = in_ci_conv == False
        if np.sum(out_ci_mask) > 0:
            ax.scatter(true_conv[out_ci_mask], np.arange(len(true_conv))[out_ci_mask],
                      alpha=0.6, s=30, color='red', marker='x', label='Out of CI (converged)')
        
        # Draw credible intervals as horizontal lines
        for i in range(len(true_conv)):
            if conv_mask[i]:
                ax.plot([ci_lower[i], ci_upper[i]], [i, i], 
                       color='blue' if in_ci[i] else 'red', alpha=0.3, linewidth=0.5)
    
    # Plot non-converged runs (light gray)
    if np.sum(nonconv_mask) > 0:
        true_nonconv = true_vals[nonconv_mask]
        nonconv_indices = np.where(nonconv_mask)[0]
        ax.scatter(true_nonconv, nonconv_indices, alpha=0.2, s=20, 
                  color='gray', marker='.', label='Non-converged')
    
    ax.set_xlabel(f'{param_name} value')
    ax.set_ylabel('Iteration')
    ax.set_title(f'{title} Calibration (n={N_ITERATIONS})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jovyan/project/demos/ddm_calibration_plot.png', dpi=150, bbox_inches='tight')
print("Saved calibration plot to: ddm_calibration_plot.png")

# Coverage bar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

if n_converged > 0:
    categories = ['Drift', 'Boundary', 'NDT', 'Average']
    coverage_values = [drift_coverage, boundary_coverage, ndt_coverage, 
                      (drift_coverage + boundary_coverage + ndt_coverage) / 3]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = ax.bar(categories, coverage_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add target line at 95%
    ax.axhline(95.0, color='red', linestyle='--', linewidth=2, label='Target (95%)')
    
    # Add value labels on bars
    for bar, val in zip(bars, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'95% Credible Interval Coverage (converged runs only, n={n_converged})')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/jovyan/project/demos/ddm_coverage_barplot.png', dpi=150, bbox_inches='tight')
    print("Saved coverage bar plot to: ddm_coverage_barplot.png")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

converged_count = n_converged
converged_pct = (converged_count / N_ITERATIONS) * 100
max_rhat_arr = np.array([r.max_rhat for r in results])
rhat_below_1_1 = np.sum(max_rhat_arr <= 1.1)
rhat_below_1_1_pct = (rhat_below_1_1 / N_ITERATIONS) * 100
iter_times = np.array([r.iteration_time for r in results])

print(f"""
Completed {N_ITERATIONS} iterations of calibration test.

Timing:
  Total time: {total_time:.2f} seconds
  Average per iteration: {np.mean(iter_times):.2f} seconds
  Fastest iteration: {np.min(iter_times):.2f} seconds
  Slowest iteration: {np.max(iter_times):.2f} seconds

Convergence:
  Converged iterations: {converged_count}/{N_ITERATIONS} ({converged_pct:.1f}%)
  Rhat <= 1.1: {rhat_below_1_1}/{N_ITERATIONS} ({rhat_below_1_1_pct:.1f}%)
  Mean Rhat: {np.mean(max_rhat_arr):.4f}
  Max Rhat: {np.max(max_rhat_arr):.4f}
  Min Rhat: {np.min(max_rhat_arr):.4f}
""")

if n_converged > 0:
    print(f"""
Calibration (converged runs only, n={n_converged}):
  Drift:    {drift_coverage:.1f}% coverage (target: 95.0%)
  Boundary: {boundary_coverage:.1f}% coverage (target: 95.0%)
  NDT:      {ndt_coverage:.1f}% coverage (target: 95.0%)
  Average:  {(drift_coverage + boundary_coverage + ndt_coverage) / 3:.1f}% coverage (target: 95.0%)
  
  Interpretation:
    - Coverage close to 95% indicates well-calibrated credible intervals
    - Coverage significantly below 95% suggests intervals are too narrow (overconfident)
    - Coverage significantly above 95% suggests intervals are too wide (underconfident)
""")
else:
    print("\nWARNING: No converged runs - cannot assess calibration.")

print("""
Plots saved:
  - ddm_calibration_plot.png
  - ddm_coverage_barplot.png
""")

