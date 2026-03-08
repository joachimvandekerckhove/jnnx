#!/usr/bin/env python3
"""
DDM Simulate-and-Recover Exercise

This script demonstrates a simulate-and-recover exercise for the DDM (Drift Diffusion Model) 
using the JAGS module.

Overview:
1. Simulate: Generate "observed" summary statistics from known true parameters using the DDM emulator
2. Recover: Use JAGS with the DDM emulator module to estimate parameters from the observed statistics
3. Compare: Evaluate how well the estimated parameters match the true parameters
"""

import py2jags
import numpy as np
import matplotlib.pyplot as plt
import time
from jnnx import JNNXPackage

# Setup and Configuration
print("=" * 60)
print("DDM Simulate-and-Recover Exercise - 200 Iterations")
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
print(f"\nRunning {N_ITERATIONS} iterations with random parameter values...")
print("Each iteration will be timed.\n")

# Storage for results
results = {
    'true_drift': [],
    'true_boundary': [],
    'true_ndt': [],
    'estimated_drift': [],
    'estimated_boundary': [],
    'estimated_ndt': [],
    'iteration_times': [],
    'errors_drift': [],
    'errors_boundary': [],
    'errors_ndt': [],
    'max_rhat': [],
    'converged': []
}

# JAGS model for parameter recovery (defined once, reused)
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
    # Using normal distributions with high precision for matching
    observed_accuracy ~ dnorm(predicted_stats[1], 1000)  # High precision for matching
    observed_mean_rt ~ dnorm(predicted_stats[2], 1000)
    observed_var_rt ~ dnorm(predicted_stats[3], 1000)
}
'''

# Run iterations
np.random.seed(42)  # For reproducibility
start_total = time.time()

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
    observed_accuracy = result.get_samples('observed_stats_1')[0]
    observed_mean_rt = result.get_samples('observed_stats_2')[0]
    observed_var_rt = result.get_samples('observed_stats_3')[0]
    
    # Step 3: Recover parameters
    recovery_model = recovery_model_template.replace('{FUNCTION_NAME}', FUNCTION_NAME)
    
    data = {
        'observed_accuracy': observed_accuracy,
        'observed_mean_rt': observed_mean_rt,
        'observed_var_rt': observed_var_rt,
        'n': 10000
    }
    
    # Run MCMC (with fewer samples for speed)
    chains = py2jags.run_jags(
        model_string=recovery_model,
        data_dict=data,
        nchains=2,
        nsamples=1000,
        nadapt=500,
        nburnin=500,
        monitorparams=['drift', 'boundary', 'ndt_scaled'],
        modules=[MODULE_NAME],
        verbosity=0
    )
    
    # Step 4: Extract estimates
    drift_samples = chains.get_samples('drift')
    boundary_samples = chains.get_samples('boundary')
    ndt_samples = chains.get_samples('ndt_scaled')
    
    drift_mean = np.mean(drift_samples)
    boundary_mean = np.mean(boundary_samples)
    ndt_mean = np.mean(ndt_samples)
    
    # Store results
    results['true_drift'].append(true_drift)
    results['true_boundary'].append(true_boundary)
    results['true_ndt'].append(true_ndt)
    results['estimated_drift'].append(drift_mean)
    results['estimated_boundary'].append(boundary_mean)
    results['estimated_ndt'].append(ndt_mean)
    results['errors_drift'].append(drift_mean - true_drift)
    results['errors_boundary'].append(boundary_mean - true_boundary)
    results['errors_ndt'].append(ndt_mean - true_ndt)
    
    iter_time = time.time() - iter_start
    results['iteration_times'].append(iter_time)
    
    # Check convergence
    max_rhat = chains.max_rhat()
    converged = chains.converged()
    results['max_rhat'].append(max_rhat)
    results['converged'].append(converged)
    
    if max_rhat > 1.1:
        print(f"Convergence warning: Max Rhat = {max_rhat:.3f} > 1.1 for iteration {iteration + 1}")
    elif not converged:
        print(f"Convergence warning: Chains not converged (Rhat = {max_rhat:.3f}) for iteration {iteration + 1}")
    
    # Progress update every 10 iterations
    if (iteration + 1) % 10 == 0:
        avg_time = np.mean(results['iteration_times'])
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
print(f"Average time per iteration: {np.mean(results['iteration_times']):.2f} seconds")
print(f"Min iteration time: {np.min(results['iteration_times']):.2f} seconds")
print(f"Max iteration time: {np.max(results['iteration_times']):.2f} seconds")
print(f"{'='*60}")

# Step 5: Analyze Results
print("\n" + "=" * 60)
print("Step 5: Analyze Results")
print("=" * 60)

# Convert to numpy arrays for easier analysis
true_drift_arr = np.array(results['true_drift'])
true_boundary_arr = np.array(results['true_boundary'])
true_ndt_arr = np.array(results['true_ndt'])
est_drift_arr = np.array(results['estimated_drift'])
est_boundary_arr = np.array(results['estimated_boundary'])
est_ndt_arr = np.array(results['estimated_ndt'])
errors_drift_arr = np.array(results['errors_drift'])
errors_boundary_arr = np.array(results['errors_boundary'])
errors_ndt_arr = np.array(results['errors_ndt'])
converged_arr = np.array(results['converged'])

# Filter to only converged runs for statistics
converged_mask = converged_arr == True
n_converged = np.sum(converged_mask)
n_non_converged = N_ITERATIONS - n_converged

print(f"\nConvergence Summary:")
print(f"  Converged: {n_converged}/{N_ITERATIONS} ({100*n_converged/N_ITERATIONS:.1f}%)")
print(f"  Non-converged: {n_non_converged}/{N_ITERATIONS} ({100*n_non_converged/N_ITERATIONS:.1f}%)")
print(f"\nRecovery statistics below are calculated ONLY for converged runs.")

# Filter arrays to only converged runs
true_drift_conv = true_drift_arr[converged_mask]
true_boundary_conv = true_boundary_arr[converged_mask]
true_ndt_conv = true_ndt_arr[converged_mask]
est_drift_conv = est_drift_arr[converged_mask]
est_boundary_conv = est_boundary_arr[converged_mask]
est_ndt_conv = est_ndt_arr[converged_mask]
errors_drift_conv = errors_drift_arr[converged_mask]
errors_boundary_conv = errors_boundary_arr[converged_mask]
errors_ndt_conv = errors_ndt_arr[converged_mask]

# Calculate summary statistics
def print_summary_stats(true_vals, est_vals, errors, param_name):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Calculate percentage errors (avoid division by zero)
    pct_errors = []
    for true, est in zip(true_vals, est_vals):
        if abs(true) > 1e-6:
            pct_errors.append(abs((est - true) / true) * 100)
        else:
            pct_errors.append(abs(est - true) * 100)
    mean_pct_error = np.mean(pct_errors)
    
    # Correlation
    correlation = np.corrcoef(true_vals, est_vals)[0, 1]
    
    print(f"\n{param_name}:")
    print(f"  Mean Error: {mean_error:.4f} ± {std_error:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean Percentage Error: {mean_pct_error:.2f}%")
    print(f"  Correlation (True vs Estimated): {correlation:.4f}")

if n_converged > 0:
    print_summary_stats(true_drift_conv, est_drift_conv, errors_drift_conv, "Drift (converged only)")
    print_summary_stats(true_boundary_conv, est_boundary_conv, errors_boundary_conv, "Boundary (converged only)")
    print_summary_stats(true_ndt_conv, est_ndt_conv, errors_ndt_conv, "NDT (converged only)")
else:
    print("\nWARNING: No converged runs! Cannot calculate recovery statistics.")

# Step 6: Visualize Results
print("\n" + "=" * 60)
print("Step 6: Visualize Results")
print("=" * 60)

# Recovery plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

param_data = [
    (true_drift_arr, est_drift_arr, 'Drift', 'drift'),
    (true_boundary_arr, est_boundary_arr, 'Boundary', 'boundary'),
    (true_ndt_arr, est_ndt_arr, 'NDT', 'ndt')
]

for idx, (true_vals, est_vals, title, param_name) in enumerate(param_data):
    ax = axes[idx]
    
    # Separate converged and non-converged
    true_conv = true_vals[converged_mask]
    est_conv = est_vals[converged_mask]
    true_nonconv = true_vals[~converged_mask]
    est_nonconv = est_vals[~converged_mask]
    
    # Plot converged runs in blue
    if len(true_conv) > 0:
        ax.scatter(true_conv, est_conv, alpha=0.6, s=25, color='blue', label=f'Converged (n={len(true_conv)})')
    
    # Plot non-converged runs in red
    if len(true_nonconv) > 0:
        ax.scatter(true_nonconv, est_nonconv, alpha=0.1, s=25, color='red', marker='x', label=f'Non-converged (n={len(true_nonconv)})')
    
    # Perfect recovery line (y=x)
    all_vals = np.concatenate([true_vals, est_vals])
    min_val = np.min(all_vals)
    max_val = np.max(all_vals)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect recovery', alpha=0.7)
    
    # Add correlation text (only for converged runs)
    if len(true_conv) > 0:
        corr = np.corrcoef(true_conv, est_conv)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}\n(n={len(true_conv)})', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'True {param_name}')
    ax.set_ylabel(f'Estimated {param_name}')
    ax.set_title(f'{title} Recovery (n={N_ITERATIONS})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jovyan/project/demos/ddm_recovery_plot_200iter.png', dpi=150, bbox_inches='tight')
print("Saved recovery plot to: ddm_recovery_plot_200iter.png")

# Error distribution plots (only converged runs)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

error_data = [
    (errors_drift_conv, 'Drift', 'drift'),
    (errors_boundary_conv, 'Boundary', 'boundary'),
    (errors_ndt_conv, 'NDT', 'ndt')
]

for idx, (errors, title, param_name) in enumerate(error_data):
    ax = axes[idx]
    
    if len(errors) > 0:
        ax.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    
    ax.set_xlabel(f'Error ({param_name})')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{title} Error Distribution (converged only, n={len(errors)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jovyan/project/demos/ddm_error_distribution_200iter.png', dpi=150, bbox_inches='tight')
print("Saved error distribution plot to: ddm_error_distribution_200iter.png")

# Timing distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(results['iteration_times'], bins=30, alpha=0.7, color='orange', edgecolor='black')
ax.axvline(np.mean(results['iteration_times']), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {np.mean(results["iteration_times"]):.2f}s')
ax.set_xlabel('Time per iteration (seconds)')
ax.set_ylabel('Frequency')
ax.set_title('Iteration Time Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/jovyan/project/demos/ddm_timing_distribution_200iter.png', dpi=150, bbox_inches='tight')
print("Saved timing distribution plot to: ddm_timing_distribution_200iter.png")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
# Convergence statistics
converged_count = sum(results['converged'])
converged_pct = (converged_count / N_ITERATIONS) * 100
max_rhat_arr = np.array(results['max_rhat'])
rhat_below_1_1 = np.sum(max_rhat_arr <= 1.1)
rhat_below_1_1_pct = (rhat_below_1_1 / N_ITERATIONS) * 100

print(f"""
Completed {N_ITERATIONS} iterations of simulate-and-recover exercise.

Timing:
  Total time: {total_time:.2f} seconds
  Average per iteration: {np.mean(results['iteration_times']):.2f} seconds
  Fastest iteration: {np.min(results['iteration_times']):.2f} seconds
  Slowest iteration: {np.max(results['iteration_times']):.2f} seconds

Convergence:
  Converged iterations: {converged_count}/{N_ITERATIONS} ({converged_pct:.1f}%)
  Rhat <= 1.1: {rhat_below_1_1}/{N_ITERATIONS} ({rhat_below_1_1_pct:.1f}%)
  Mean Rhat: {np.mean(max_rhat_arr):.4f}
  Max Rhat: {np.max(max_rhat_arr):.4f}
  Min Rhat: {np.min(max_rhat_arr):.4f}

Recovery Quality (converged runs only, n={n_converged}):""")
if n_converged > 0:
    print(f"""  Drift:    MAE={np.mean(np.abs(errors_drift_conv)):.4f},    r={np.corrcoef(true_drift_conv, est_drift_conv)[0,1]:.4f}
  Boundary: MAE={np.mean(np.abs(errors_boundary_conv)):.4f}, r={np.corrcoef(true_boundary_conv, est_boundary_conv)[0,1]:.4f}
  NDT:      MAE={np.mean(np.abs(errors_ndt_conv)):.4f},      r={np.corrcoef(true_ndt_conv, est_ndt_conv)[0,1]:.4f}""")
else:
    print("  No converged runs - cannot calculate recovery quality.")

print("""
Plots saved:
  - ddm_recovery_plot_200iter.png
  - ddm_error_distribution_200iter.png
  - ddm_timing_distribution_200iter.png
""")

