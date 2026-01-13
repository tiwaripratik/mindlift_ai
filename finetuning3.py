# -*- coding: utf-8 -*-
"""
Fine-tuning script for Mistral 7B to learn optimal weight parameter (w) 
for Profit-SLA Violation (PSV) optimization using PlanetLab workload traces.

This script:
1. Loads PlanetLab CPU utilization traces from VM workload data
2. Calculates profit and SLA violations from workload patterns
3. Determines optimal weight w for PSV = w * profit + (1-w) * SLAV
4. Creates input/output pairs for training
5. Fine-tunes Mistral 7B using LoRA to predict optimal w values

PSV Formula:
PSV = w * profit + (1-w) * SLAV

Where:
- w: weight parameter (0-1) to be learned by the model
- profit: revenue/benefit from resource utilization
- SLAV: Service Level Agreement Violations (to be minimized)

COLAB SETUP INSTRUCTIONS:
=========================
OPTION 1 (Recommended): Install packages first, then run script
---------------------------------------------------------------
# Run this in a Colab cell first:
!pip install -q transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch numpy scipy

# Download PlanetLab dataset:
# Option 1: Clone the repository
!git clone https://github.com/beloglazov/planetlab-workload-traces.git

# Option 2: Download ZIP and upload to Colab (script will auto-extract)
# 1. Download ZIP from GitHub
# 2. Upload to Colab: from google.colab import files; files.upload()
# 3. The script will automatically detect and extract the ZIP file

# Then run this script:
!python finetuning3.py

OPTION 2: Let script auto-install (may take longer)
---------------------------------------------------
# Just copy/paste this entire file into a Colab cell and run it.
# The script will automatically install missing packages.
"""

import os
import sys
import subprocess
import json
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import minimize_scalar

# Set PyTorch CUDA memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Quantization setting - MUST be defined before imports to avoid bitsandbytes issues
USE_4BIT = False  # Set to False to disable quantization and avoid bitsandbytes issues (RECOMMENDED)
# USE_4BIT = True  # Only enable if bitsandbytes is working properly

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install required packages if not already installed (do this BEFORE importing)
if IN_COLAB:
    print("Running in Google Colab environment")
    print("Checking and installing required packages...")
    
    required_packages = [
        ('transformers', '>=4.35.0'),
        ('datasets', '>=2.15.0'),
        ('peft', '>=0.7.0'),
        ('bitsandbytes', '>=0.41.0'),
        ('trl', '>=0.7.0'),
        ('accelerate', '>=0.25.0'),
        ('pandas', '>=2.0.0'),
        ('torch', '>=2.0.0'),
        ('numpy', '>=1.24.0'),
        ('scipy', '>=1.10.0'),
    ]
    
    # Special handling for torch and bitsandbytes due to compatibility issues
    # We'll skip these import checks to avoid AttributeError/RuntimeError
    torch_handled = False
    bitsandbytes_handled = False
    
    for package_name, version in required_packages:
        # Skip torch and bitsandbytes in the main loop, handle them separately
        # These can cause RuntimeError/AttributeError during import checks
        if package_name == 'torch':
            torch_handled = True
            continue
        if package_name == 'bitsandbytes':
            bitsandbytes_handled = True
            continue
        
        try:
            # Try to import to check if it's installed
            __import__(package_name)
            continue  # Package is installed, skip installation
        except ImportError:
            pass  # Package not installed, will install below
        except (AttributeError, RuntimeError) as e:
            # Handle AttributeError from package imports (e.g., sympy issues)
            # Handle RuntimeError from bitsandbytes/PyTorch conflicts
            error_str = str(e).lower()
            if 'sympy' in error_str or 'printing' in error_str:
                print(f"⚠ Warning: Detected compatibility issue with {package_name} during import check")
                print(f"   This is usually harmless. Package will be verified during actual import.")
                continue  # Skip installation, assume package is installed
            elif 'bitsandbytes' in error_str or 'kernel registered' in error_str:
                print(f"⚠ Warning: Detected bitsandbytes/PyTorch conflict with {package_name} during import check")
                print(f"   This is usually harmless. Package will be verified during actual import.")
                continue  # Skip installation, assume package is installed
            raise  # Re-raise if it's a different error
        
        # Package not found, try to install it
        print(f"Installing {package_name}{version}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}{version}", "-q", "--upgrade"
            ])
            print(f"✓ Installed {package_name}")
        except Exception as e:
            print(f"⚠ Warning: Could not install {package_name}: {e}")
            print(f"Please install manually: !pip install {package_name}{version}")
    
    # Handle torch separately - skip import check to avoid sympy AttributeError
    # Torch will be verified during actual import later in the script
    if torch_handled:
        print("⚠ Skipping torch import check (due to sympy compatibility)")
        print("   torch will be verified during actual import. If missing, install with: !pip install torch>=2.0.0")
    
    # Handle bitsandbytes separately - skip import check to avoid RuntimeError conflicts
    # bitsandbytes will be verified during actual import later in the script (if USE_4BIT is True)
    if bitsandbytes_handled:
        print("⚠ Skipping bitsandbytes import check (due to PyTorch compatibility conflicts)")
        print("   bitsandbytes will be verified during actual import if quantization is enabled.")
    
    print("Package check complete!\n")

# Now import the packages - wrap in try/except for better error messages
try:
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        pipeline,
        logging,
    )
    # Only import bitsandbytes-related modules if USE_4BIT is True
    BitsAndBytesConfig = None
    prepare_model_for_kbit_training = None
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    else:
        from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
except ImportError as e:
    print("=" * 80)
    print("ERROR: Missing or incompatible packages!")
    print("=" * 80)
    error_str = str(e).lower()
    
    # Check for specific transformers version issues
    if 'generationmixin' in error_str or 'generation' in error_str:
        print("\n⚠️  Transformers version incompatibility detected!")
        print("   This usually happens when transformers version is too old or incompatible.")
        print("\nSOLUTION:")
        print("   Please upgrade transformers:")
        print("   !pip install --upgrade transformers>=4.35.0")
        print("\n   Or reinstall all packages:")
        print("   !pip install --upgrade transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 trl>=0.7.0 accelerate>=0.25.0")
    else:
        print(f"\nFailed to import: {e}")
        print("\nPlease install required packages:")
        print("!pip install transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch numpy scipy")
    
    print("\nOr run this in a Colab cell first:")
    print("!pip install -q --upgrade transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch numpy scipy")
    print("=" * 80)
    
    # Try to auto-fix by upgrading transformers
    if IN_COLAB and ('generationmixin' in error_str or 'generation' in error_str):
        print("\n🔄 Attempting to auto-fix by upgrading transformers...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "transformers>=4.35.0", "-q"
            ])
            print("✓ Upgraded transformers. Please restart the runtime and run again:")
            print("  Runtime -> Restart runtime")
            print("  Then run this script again.")
        except Exception as upgrade_error:
            print(f"⚠️  Auto-fix failed: {upgrade_error}")
            print("   Please manually upgrade transformers as shown above.")
    
    # Use raise SystemExit instead of sys.exit() to avoid IPython traceback issues
    raise SystemExit(1)
except Exception as e:
    print("=" * 80)
    print("ERROR: Unexpected error during package import!")
    print("=" * 80)
    print(f"Error: {e}")
    print("\nThis might be a version compatibility issue.")
    print("Try upgrading packages:")
    print("!pip install --upgrade transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 trl>=0.7.0 accelerate>=0.25.0")
    print("=" * 80)
    # Use raise SystemExit instead of sys.exit() to avoid IPython traceback issues
    raise SystemExit(1)


# Configuration parameters for profit and SLAV calculation
PROFIT_PER_UNIT_UTILIZATION = 1.0  # Revenue per unit of CPU utilization
SLA_THRESHOLD = 80.0  # CPU utilization threshold (%) for SLA violation
SLA_PENALTY_PER_VIOLATION = 10.0  # Penalty cost per SLA violation
MIN_UTILIZATION_FOR_PROFIT = 10.0  # Minimum utilization to generate profit


def load_planetlab_traces(dataset_dir: str) -> List[Dict]:
    """
    Load PlanetLab workload traces from directory structure.
    
    Expected structure:
    dataset_dir/
        YYYYMMDD/
            *.txt (VM trace files)
    
    Each trace file contains CPU utilization percentages (0-100) per line.
    Data is sampled every 5 minutes.
    
    Returns:
        List of dictionaries with VM trace data and metadata
    """
    traces = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all date directories (format: YYYYMMDD)
    date_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit() and len(d) == 8]
    
    if not date_dirs:
        raise ValueError(f"No date directories found in {dataset_dir}. Expected format: YYYYMMDD/")
    
    print(f"   Found {len(date_dirs)} date directories")
    
    for date_dir in sorted(date_dirs):
        date_path = os.path.join(dataset_dir, date_dir)
        
        # Find all trace files in this date directory
        trace_files = glob.glob(os.path.join(date_path, "*.txt"))
        
        if not trace_files:
            # Try .csv files
            trace_files = glob.glob(os.path.join(date_path, "*.csv"))
        
        if not trace_files:
            # Try any files
            trace_files = [f for f in os.listdir(date_path) 
                          if os.path.isfile(os.path.join(date_path, f))]
        
        print(f"   Date {date_dir}: Found {len(trace_files)} trace files")
        
        for trace_file in trace_files:
            trace_path = os.path.join(date_path, trace_file)
            
            try:
                # Read CPU utilization values
                # PlanetLab traces are typically space-separated or one value per line
                with open(trace_path, 'r') as f:
                    lines = f.readlines()
                
                # Parse CPU utilization values
                cpu_values = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Try to extract numeric values
                    parts = line.split()
                    for part in parts:
                        try:
                            value = float(part)
                            # Normalize to 0-100 range if needed
                            if value > 100:
                                value = min(value, 100.0)
                            elif value < 0:
                                value = max(value, 0.0)
                            cpu_values.append(value)
                        except ValueError:
                            continue
                
                if len(cpu_values) < 10:  # Skip files with too few data points
                    continue
                
                # Store trace data
                traces.append({
                    'date': date_dir,
                    'vm_id': os.path.basename(trace_file),
                    'cpu_utilization': np.array(cpu_values),
                    'trace_file': trace_path
                })
                
            except Exception as e:
                print(f"   ⚠ Warning: Could not read {trace_file}: {e}")
                continue
    
    print(f"   ✓ Loaded {len(traces)} VM traces")
    return traces


def calculate_profit(cpu_utilization: np.ndarray) -> float:
    """
    Calculate profit from CPU utilization.
    
    Profit is based on:
    - Higher utilization = more revenue (up to a point)
    - But excessive utilization may indicate overload
    
    Args:
        cpu_utilization: Array of CPU utilization percentages
        
    Returns:
        Profit value
    """
    if len(cpu_utilization) == 0:
        return 0.0
    
    # Profit increases with utilization, but with diminishing returns
    # Optimal utilization is around 70-80%
    avg_utilization = np.mean(cpu_utilization)
    
    # Base profit from average utilization
    base_profit = avg_utilization * PROFIT_PER_UNIT_UTILIZATION
    
    # Bonus for stable utilization (lower variance = more predictable = better)
    utilization_std = np.std(cpu_utilization)
    stability_bonus = max(0, (100 - utilization_std) / 100) * 10
    
    # Penalty for very high utilization (indicates overload risk)
    if avg_utilization > 90:
        overload_penalty = (avg_utilization - 90) * 2
    else:
        overload_penalty = 0
    
    profit = base_profit + stability_bonus - overload_penalty
    
    # Ensure profit is non-negative
    return max(0.0, profit)


def calculate_slav(cpu_utilization: np.ndarray) -> float:
    """
    Calculate SLA Violations (SLAV) from CPU utilization.
    
    SLA violations occur when:
    - CPU utilization exceeds SLA threshold (over-provisioning)
    - CPU utilization is too low (under-utilization waste)
    
    Args:
        cpu_utilization: Array of CPU utilization percentages
        
    Returns:
        SLAV value (lower is better)
    """
    if len(cpu_utilization) == 0:
        return 100.0  # Maximum penalty for no data
    
    # Count violations above SLA threshold
    violations_above = np.sum(cpu_utilization > SLA_THRESHOLD)
    
    # Count violations below minimum utilization (waste)
    violations_below = np.sum(cpu_utilization < MIN_UTILIZATION_FOR_PROFIT)
    
    # Total violations
    total_violations = violations_above + violations_below
    
    # Calculate SLAV as normalized violation rate
    violation_rate = total_violations / len(cpu_utilization)
    
    # Scale to make it comparable with profit
    slav = violation_rate * SLA_PENALTY_PER_VIOLATION * 100
    
    # Also consider severity (how far from threshold)
    if violations_above > 0:
        avg_excess = np.mean(cpu_utilization[cpu_utilization > SLA_THRESHOLD] - SLA_THRESHOLD)
        slav += avg_excess * 0.5
    
    return slav


def find_optimal_weight(profits: List[float], slavs: List[float]) -> float:
    """
    Find optimal weight w that maximizes PSV = w * profit + (1-w) * SLAV.
    
    Since we want to maximize profit and minimize SLAV, and SLAV is a penalty
    (lower is better), we need to transform the formula to:
    PSV = w * profit - (1-w) * SLAV
    
    However, the user specified: PSV = w * profit + (1-w) * SLAV
    This means we want to balance profit maximization (weight w) with 
    SLAV minimization (weight 1-w). To maximize PSV:
    - When w=1: Focus entirely on profit
    - When w=0: Focus entirely on minimizing SLAV (through negative SLAV)
    
    For the optimization, we'll use: PSV = w * profit - (1-w) * SLAV
    to properly account for minimizing SLAV.
    
    Args:
        profits: List of profit values (higher is better)
        slavs: List of SLAV values (lower is better)
        
    Returns:
        Optimal weight w (0-1)
    """
    if len(profits) != len(slavs) or len(profits) == 0:
        return 0.5  # Default weight
    
    profits_array = np.array(profits)
    slavs_array = np.array(slavs)
    
    # Normalize both to 0-1 range for fair comparison
    max_profit = np.max(profits_array) if np.max(profits_array) > 0 else 1.0
    min_profit = np.min(profits_array)
    profit_range = max_profit - min_profit if max_profit > min_profit else 1.0
    normalized_profit = (profits_array - min_profit) / profit_range
    
    max_slav = np.max(slavs_array) if np.max(slavs_array) > 0 else 1.0
    min_slav = np.min(slavs_array)
    slav_range = max_slav - min_slav if max_slav > min_slav else 1.0
    normalized_slav = (slavs_array - min_slav) / slav_range
    
    # Objective: maximize PSV = w * profit - (1-w) * SLAV
    # We subtract SLAV because lower SLAV is better
    def objective(w):
        # Ensure w is in [0, 1]
        w = max(0.0, min(1.0, w))
        psv = w * normalized_profit - (1 - w) * normalized_slav
        return -np.mean(psv)  # Negative because minimize_scalar minimizes
    
    # Find optimal w using bounded optimization
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    
    optimal_w = result.x if result.success else 0.5
    
    return max(0.0, min(1.0, optimal_w))


def extract_workload_features(cpu_utilization: np.ndarray) -> Dict:
    """
    Extract features from workload trace for model input.
    
    Args:
        cpu_utilization: Array of CPU utilization percentages
        
    Returns:
        Dictionary of workload features
    """
    if len(cpu_utilization) == 0:
        return {}
    
    features = {
        'mean_utilization': float(np.mean(cpu_utilization)),
        'std_utilization': float(np.std(cpu_utilization)),
        'min_utilization': float(np.min(cpu_utilization)),
        'max_utilization': float(np.max(cpu_utilization)),
        'median_utilization': float(np.median(cpu_utilization)),
        'q25_utilization': float(np.percentile(cpu_utilization, 25)),
        'q75_utilization': float(np.percentile(cpu_utilization, 75)),
        'cv_utilization': float(np.std(cpu_utilization) / np.mean(cpu_utilization)) if np.mean(cpu_utilization) > 0 else 0.0,
        'samples_count': len(cpu_utilization),
        'violations_above_threshold': int(np.sum(cpu_utilization > SLA_THRESHOLD)),
        'violations_below_minimum': int(np.sum(cpu_utilization < MIN_UTILIZATION_FOR_PROFIT)),
        'utilization_range': float(np.max(cpu_utilization) - np.min(cpu_utilization)),
    }
    
    return features


def process_traces_to_training_pairs(traces: List[Dict]) -> List[Dict]:
    """
    Process PlanetLab traces into training pairs for weight prediction.
    
    Format:
    - Input: Workload features and statistics
    - Output: Optimal weight w for PSV optimization
    
    Args:
        traces: List of trace dictionaries
        
    Returns:
        List of training pairs
    """
    training_pairs = []
    
    # Calculate profit and SLAV for all traces
    all_profits = []
    all_slavs = []
    trace_features = []
    
    print("\n   Calculating profit and SLAV for traces...")
    for trace in traces:
        cpu_util = trace['cpu_utilization']
        profit = calculate_profit(cpu_util)
        slav = calculate_slav(cpu_util)
        features = extract_workload_features(cpu_util)
        
        all_profits.append(profit)
        all_slavs.append(slav)
        trace_features.append({
            **trace,
            'profit': profit,
            'slav': slav,
            'features': features
        })
    
    print(f"   Calculated metrics for {len(trace_features)} traces")
    print(f"   Profit range: {min(all_profits):.2f} - {max(all_profits):.2f}")
    print(f"   SLAV range: {min(all_slavs):.2f} - {max(all_slavs):.2f}")
    
    # Find optimal weight for the entire dataset
    global_optimal_w = find_optimal_weight(all_profits, all_slavs)
    print(f"   Global optimal weight w: {global_optimal_w:.4f}")
    
    # Create training pairs
    print("\n   Creating training pairs...")
    
    # Method 1: Predict optimal w for individual traces
    for trace_data in trace_features:
        features = trace_data['features']
        profit = trace_data['profit']
        slav = trace_data['slav']
        
        # Find optimal w for this specific trace (considering it in context)
        # We'll use a local optimization approach
        optimal_w = find_optimal_weight([profit], [slav])
        
        # Create input text describing the workload
        input_text = f"""Analyze the following VM workload trace and determine the optimal weight parameter (w) 
for the Profit-SLA Violation (PSV) optimization formula:

PSV = w * profit + (1-w) * SLAV

Where:
- w: weight parameter (0-1) to balance profit maximization vs SLA violation minimization
- profit: revenue/benefit from resource utilization (higher is better)
- SLAV: Service Level Agreement Violations (lower is better, treated as penalty)

Objective: Maximize PSV by balancing:
- Profit maximization (weight w): Higher profit increases PSV
- SLAV minimization (weight 1-w): Lower SLAV is better, so lower (1-w)*SLAV term

Workload Statistics:
- Mean CPU Utilization: {features['mean_utilization']:.2f}%
- Standard Deviation: {features['std_utilization']:.2f}%
- Min Utilization: {features['min_utilization']:.2f}%
- Max Utilization: {features['max_utilization']:.2f}%
- Median Utilization: {features['median_utilization']:.2f}%
- Coefficient of Variation: {features['cv_utilization']:.4f}
- Utilization Range: {features['utilization_range']:.2f}%
- Violations Above Threshold ({SLA_THRESHOLD}%): {features['violations_above_threshold']}
- Violations Below Minimum ({MIN_UTILIZATION_FOR_PROFIT}%): {features['violations_below_minimum']}
- Total Samples: {features['samples_count']}

Calculated Metrics:
- Profit: {profit:.4f}
- SLAV: {slav:.4f}

Determine the optimal weight w that maximizes PSV while balancing profit maximization and SLA violation minimization."""
        
        # Format ratio safely
        if slav > 0:
            ratio_str = f'{profit/slav:.4f}'
        else:
            ratio_str = 'N/A'
        
        output_text = """Optimal Weight Parameter (w): {:.6f}

Reasoning:
- Profit value: {:.4f}
- SLAV value: {:.4f}
- Profit-to-SLAV ratio: {}

The optimal weight w = {:.6f} balances the trade-off between:
1. Maximizing profit (weight: {:.4f})
2. Minimizing SLA violations (weight: {:.4f})

PSV Calculation (using formula: PSV = w * profit + (1-w) * SLAV):
PSV = {:.6f} * {:.4f} + (1 - {:.6f}) * {:.4f}
PSV = {:.4f}

Note: Since SLAV is a penalty term (lower is better), the optimal w is chosen to maximize PSV 
by appropriately weighting profit maximization against SLAV minimization.

Interpretation:
- When w approaches 1.0: Focus on maximizing profit
- When w approaches 0.0: Focus on minimizing SLA violations
- Optimal w = {:.6f}: Balanced optimization strategy""".format(
            optimal_w, profit, slav, ratio_str, optimal_w, optimal_w, 1-optimal_w,
            optimal_w, profit, optimal_w, slav, optimal_w * profit + (1 - optimal_w) * slav,
            optimal_w
        )
        
        training_pairs.append({
            "input": input_text,
            "output": output_text,
            "profit": profit,
            "slav": slav,
            "optimal_w": optimal_w,
            "features": features
        })
    
    # Method 2: Create pairs for batch/aggregate scenarios
    # Group traces by date and create aggregate training pairs
    traces_by_date = {}
    for trace_data in trace_features:
        date = trace_data['date']
        if date not in traces_by_date:
            traces_by_date[date] = []
        traces_by_date[date].append(trace_data)
    
    print(f"\n   Creating aggregate training pairs for {len(traces_by_date)} dates...")
    
    for date, date_traces in traces_by_date.items():
        if len(date_traces) < 2:
            continue
        
        # Aggregate features
        date_profits = [t['profit'] for t in date_traces]
        date_slavs = [t['slav'] for t in date_traces]
        
        # Aggregate statistics
        avg_mean_util = np.mean([t['features']['mean_utilization'] for t in date_traces])
        avg_std_util = np.mean([t['features']['std_utilization'] for t in date_traces])
        total_violations = sum([t['features']['violations_above_threshold'] + t['features']['violations_below_minimum'] 
                               for t in date_traces])
        
        # Find optimal w for this date's aggregate
        optimal_w = find_optimal_weight(date_profits, date_slavs)
        avg_profit = np.mean(date_profits)
        avg_slav = np.mean(date_slavs)
        
        input_text = f"""Analyze the following aggregated VM workload data from date {date} and determine 
the optimal weight parameter (w) for the Profit-SLA Violation (PSV) optimization formula:

PSV = w * profit + (1-w) * SLAV

Where:
- w: weight parameter (0-1) to balance profit maximization vs SLA violation minimization
- profit: revenue/benefit from resource utilization (higher is better)
- SLAV: Service Level Agreement Violations (lower is better, treated as penalty)

Objective: Maximize PSV by balancing profit maximization and SLAV minimization.

Aggregate Workload Statistics:
- Number of VMs: {len(date_traces)}
- Average Mean CPU Utilization: {avg_mean_util:.2f}%
- Average Standard Deviation: {avg_std_util:.2f}%
- Total SLA Violations: {total_violations}

Aggregate Metrics:
- Average Profit: {avg_profit:.4f}
- Average SLAV: {avg_slav:.4f}

Determine the optimal weight w that maximizes PSV for this aggregated workload."""
        
        # Format ratio safely
        if avg_slav > 0:
            ratio_str = f'{avg_profit/avg_slav:.4f}'
        else:
            ratio_str = 'N/A'
        
        output_text = f"""Optimal Weight Parameter (w): {optimal_w:.6f}

Reasoning:
- Average Profit: {avg_profit:.4f}
- Average SLAV: {avg_slav:.4f}
- Profit-to-SLAV ratio: {ratio_str}

The optimal weight w = {optimal_w:.6f} balances the trade-off between:
1. Maximizing profit (weight: {optimal_w:.4f})
2. Minimizing SLA violations (weight: {1-optimal_w:.4f})

PSV Calculation:
PSV = {optimal_w:.6f} * {avg_profit:.4f} + (1 - {optimal_w:.6f}) * {avg_slav:.4f}
PSV = {optimal_w * avg_profit + (1 - optimal_w) * avg_slav:.4f}

Interpretation:
- When w approaches 1.0: Focus on maximizing profit
- When w approaches 0.0: Focus on minimizing SLA violations
- Optimal w = {optimal_w:.6f}: Balanced optimization strategy for {len(date_traces)} VMs"""
        
        training_pairs.append({
            "input": input_text,
            "output": output_text,
            "profit": avg_profit,
            "slav": avg_slav,
            "optimal_w": optimal_w,
            "features": {
                'avg_mean_utilization': avg_mean_util,
                'avg_std_utilization': avg_std_util,
                'total_violations': total_violations,
                'vm_count': len(date_traces)
            }
        })
    
    print(f"   ✓ Created {len(training_pairs)} training pairs")
    print(f"   - Individual trace pairs: {len(trace_features)}")
    print(f"   - Aggregate pairs: {len(training_pairs) - len(trace_features)}")
    
    return training_pairs


def save_training_data(training_pairs: List[Dict], output_path: str):
    """Save training data to JSON file"""
    # Remove numpy arrays and other non-serializable objects for JSON
    serializable_pairs = []
    for pair in training_pairs:
        serializable_pair = {
            "input": pair["input"],
            "output": pair["output"],
            "profit": float(pair["profit"]),
            "slav": float(pair["slav"]),
            "optimal_w": float(pair["optimal_w"]),
        }
        serializable_pairs.append(serializable_pair)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(serializable_pairs)} training pairs to {output_path}")


# LoRA parameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Quantization parameters (USE_4BIT is defined at top of file)
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# Training parameters
OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 18
PER_DEVICE_EVAL_BATCH_SIZE = 18
GRADIENT_ACCUMULATION_STEPS = 3
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = False
SAVE_STEPS = 500
LOGGING_STEPS = 25
MAX_SEQ_LENGTH = 2048
PACKING = False


def formatting_func(example):
    """Format training examples for Mistral"""
    return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"


def main():
    """Main training function - COLAB COMPATIBLE"""
    print("=" * 80)
    print("Mistral 7B Fine-tuning for PSV Weight Optimization")
    print("Using PlanetLab Workload Traces")
    print("=" * 80)

    # Warn if quantization is enabled
    if USE_4BIT:
        print("\n⚠️  WARNING: Quantization is ENABLED (USE_4BIT = True)")
        print("If you encounter bitsandbytes errors, set USE_4BIT = False at line 52")
        print("=" * 80)
    else:
        print("\n✓ Quantization is DISABLED (USE_4BIT = False) - bitsandbytes will not be used")
        print("=" * 80)

    # Step 1: Load PlanetLab dataset
    dataset_dir = None
    possible_paths = [
        "planetlab-workload-traces-master",  # ZIP download name
        "./planetlab-workload-traces-master",
        "planetlab-workload-traces",  # Git clone name
        "./planetlab-workload-traces",
        "/content/planetlab-workload-traces-master",
        "/content/planetlab-workload-traces",
        "/content/drive/MyDrive/planetlab-workload-traces-master",
        "/content/drive/MyDrive/planetlab-workload-traces",
    ]
    
    # First, check if dataset directory exists
    for path in possible_paths:
        if os.path.exists(path):
            dataset_dir = path
            break
    
    # If not found, check for ZIP files in Colab
    if dataset_dir is None and IN_COLAB:
        print("\n   Checking for ZIP files...")
        import zipfile
        
        zip_paths = [
            "planetlab-workload-traces-master.zip",
            "/content/planetlab-workload-traces-master.zip",
            "/content/drive/MyDrive/planetlab-workload-traces-master.zip",
        ]
        
        for zip_path in zip_paths:
            if os.path.exists(zip_path):
                print(f"   Found ZIP file: {zip_path}")
                print(f"   Extracting to /content/...")
                try:
                    extract_dir = "/content"
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"   ✓ Extracted successfully")
                    
                    # Check what was extracted
                    extracted_dirs = [d for d in os.listdir(extract_dir) 
                                     if os.path.isdir(os.path.join(extract_dir, d)) 
                                     and "planetlab" in d.lower()]
                    
                    if extracted_dirs:
                        dataset_dir = os.path.join(extract_dir, extracted_dirs[0])
                        print(f"   ✓ Found extracted dataset: {dataset_dir}")
                        break
                    else:
                        print(f"   ⚠️  Extracted but couldn't find dataset directory")
                except Exception as e:
                    print(f"   ⚠️  Failed to extract ZIP: {e}")
                    continue
    
    # If still not found, search for any planetlab directories
    if dataset_dir is None:
        print("\n⚠️  PlanetLab dataset directory not found in common locations.")
        print("Please download the dataset:")
        print("  git clone https://github.com/beloglazov/planetlab-workload-traces.git")
        print("  OR download ZIP and extract as 'planetlab-workload-traces-master'")
        print("\nOr specify the dataset directory path.")
        print("\nTrying to find dataset directories...")
        try:
            # Check current directory
            if os.path.exists("."):
                dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
                planetlab_dirs = [d for d in dirs if "planetlab" in d.lower()]
                if planetlab_dirs:
                    dataset_dir = planetlab_dirs[0]
                    print(f"Found in current directory: {dataset_dir}")
            
            # Check /content (Colab)
            if dataset_dir is None and os.path.exists("/content"):
                dirs = [d for d in os.listdir("/content") if os.path.isdir(os.path.join("/content", d))]
                planetlab_dirs = [d for d in dirs if "planetlab" in d.lower()]
                if planetlab_dirs:
                    dataset_dir = f"/content/{planetlab_dirs[0]}"
                    print(f"Found in /content: {dataset_dir}")
            
            if dataset_dir is None:
                print("No PlanetLab directories found. Please download the dataset.")
                return
        except Exception as e:
            print(f"Error checking directories: {e}")
            return
    
    print(f"\n1. Loading PlanetLab traces from {dataset_dir}...")
    try:
        traces = load_planetlab_traces(dataset_dir)
        print(f"   ✓ Loaded {len(traces)} VM traces")
    except Exception as e:
        print(f"   ERROR: Failed to load traces: {e}")
        return
    
    if len(traces) == 0:
        print("ERROR: No traces loaded. Please check the dataset directory structure.")
        return
    
    # Step 2: Process into training pairs
    print("\n2. Processing traces into training pairs...")
    training_pairs = process_traces_to_training_pairs(traces)
    print(f"   ✓ Created {len(training_pairs)} training pairs")
    
    if len(training_pairs) == 0:
        print("ERROR: No training pairs created. Please check your data.")
        return
    
    # Step 3: Save training data
    training_data_path = "psv_training_data.json"
    print(f"\n3. Saving training data to {training_data_path}...")
    save_training_data(training_pairs, training_data_path)
    
    # Step 4: Load model and tokenizer
    print("\n4. Loading Mistral 7B model and tokenizer...")
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    device_map = "cpu" if not torch.cuda.is_available() else "auto"
    print(f"   Using device: {device_map}")
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("   ✓ Cleared GPU memory cache")
    
    # Check if quantization is disabled
    use_quantization = False
    bitsandbytes_works = False
    
    if not USE_4BIT:
        print("   Quantization disabled (USE_4BIT = False) - loading model without quantization")
    else:
        print("   Testing bitsandbytes compatibility...")
        try:
            import bitsandbytes as bnb
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.randn(2, 2, dtype=torch.float16).cuda()
                    version = bnb.__version__
                    print(f"   Found bitsandbytes version: {version}")
                    bitsandbytes_works = True
                    use_quantization = True
                    print("   ✓ bitsandbytes appears compatible - will use quantization")
                except Exception as e:
                    print(f"   ⚠️  bitsandbytes CUDA functions failed: {e}")
                    bitsandbytes_works = False
                    use_quantization = False
            else:
                print("   ⚠️  No CUDA available, quantization disabled")
                bitsandbytes_works = False
                use_quantization = False
        except Exception as e:
            print(f"   ⚠️  bitsandbytes test failed: {e}")
            bitsandbytes_works = False
            use_quantization = False
        
        if not bitsandbytes_works:
            print("   ⚠️  bitsandbytes not working, quantization DISABLED")
            print("   💡 The model will use more GPU memory but will work fine")
    
    model = None
    
    # Try to load with quantization only if bitsandbytes works AND USE_4BIT is True
    if USE_4BIT and use_quantization and bitsandbytes_works:
        try:
            print("   Attempting to load with 4-bit quantization...")
            compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=USE_NESTED_QUANT,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model = prepare_model_for_kbit_training(model)
            print("   ✓ Successfully loaded with 4-bit quantization")
        except RuntimeError as e:
            if "bitsandbytes" in str(e).lower() or "cuda" in str(e).lower():
                print("   ⚠️  Quantization failed during model loading")
                print("   Falling back to loading without quantization...")
                use_quantization = False
                bitsandbytes_works = False
                model = None
            else:
                raise
    
    # Load without quantization if quantization failed or wasn't requested
    if model is None:
        print("   Loading model without quantization...")
        
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            print(f"   Available GPU memory: {free_memory_gb:.2f} GB")
            
            if free_memory_gb < 12:
                print("   ⚠️  WARNING: Low GPU memory detected!")
                print("   Consider using a smaller model or enabling quantization.")
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        if GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        print("   ✓ Successfully loaded model without quantization")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Step 5: Configure LoRA
    print("\n5. Configuring LoRA...")
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Step 6: Create dataset
    print("\n6. Creating training dataset...")
    dataset = Dataset.from_list(training_pairs)
    print(f"   Dataset size: {len(dataset)}")
    
    # Step 7: Configure training arguments
    print("\n7. Configuring training arguments...")
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        bf16=False,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to=["tensorboard"] if IN_COLAB else [],
        remove_unused_columns=False,
        disable_tqdm=False,
        logging_first_step=True,
        prediction_loss_only=False,
        load_best_model_at_end=False,
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=1,
    )
    
    # Step 8: Create trainer
    print("\n8. Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        args=training_arguments,
    )

    # Step 9: Train model
    print("\n9. Starting training...")
    print("=" * 80)
    
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        print(f"\n💾 GPU Memory Status:")
        print(f"   Total GPU Memory: {total_memory:.1f} GB")
        print(f"   Allocated: {allocated_memory:.1f} GB")
        print(f"   Reserved: {reserved_memory:.1f} GB")
        print(f"   Free: {free_memory:.1f} GB")
        print(f"\n⚙️  Training Configuration:")
        print(f"   Batch Size: {PER_DEVICE_TRAIN_BATCH_SIZE} (per device)")
        print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Max Sequence Length: {MAX_SEQ_LENGTH}")
        print(f"   Mixed Precision (fp16): True")
        print(f"   Gradient Checkpointing: {GRADIENT_CHECKPOINTING}")
    
    print("\n📊 Training progress will be displayed below:")
    print("=" * 80 + "\n")
    
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("   ✓ GPU memory cleared before training")
    
    try:
        trainer.train()
    except RuntimeError as e:
        error_str = str(e).lower()
        if ("bitsandbytes" in error_str or "cuda" in error_str or 
            "lib.cget_managed_ptr" in error_str):
            print("\n" + "=" * 80)
            print("ERROR: bitsandbytes failed during training")
            print("=" * 80)
            print("\nAutomatically recovering by reloading model without quantization...")
            
            import gc
            try:
                del trainer
                del model
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            os.environ['BITSANDBYTES_NOWELCOME'] = '1'
            
            print("   Reloading model WITHOUT quantization...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                model.config.use_cache = False
                model.config.pretraining_tp = 1
                
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    formatting_func=formatting_func,
                    processing_class=tokenizer,
                    args=training_arguments,
                )
                
                print("   ✓ Model reloaded without quantization")
                print("   Retrying training without quantization...")
                print("=" * 80)
                
                trainer.train()
            except RuntimeError as e2:
                print("\n" + "=" * 80)
                print("ERROR: Recovery failed")
                print("=" * 80)
                print("\nSOLUTION:")
                print("1. Set USE_4BIT = False at line 52 of the script")
                print("2. Restart the runtime")
                print("3. Run the script again")
                print("=" * 80)
                raise RuntimeError("bitsandbytes error - Please set USE_4BIT = False") from e2
        else:
            raise
    
    # Step 10: Save model
    print("\n10. Saving fine-tuned model...")
    trainer.model.save_pretrained("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("   Model saved to ./final_model")
    
    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    
    # Colab download helper
    if IN_COLAB:
        print("\n" + "=" * 80)
        print("COLAB: To download your trained model, run:")
        print("=" * 80)
        print("""
from google.colab import files
import shutil
import os

if os.path.exists('./final_model'):
    shutil.make_archive('final_model', 'zip', './final_model')
    files.download('final_model.zip')
    print("Model download started!")
else:
    print("Model directory not found.")
        """)
    
    # Test the model
    print("\nTesting the model...")
    logging.set_verbosity(logging.CRITICAL)
    
    prompt = """Analyze the following VM workload trace and determine the optimal weight parameter (w) 
for the Profit-SLA Violation (PSV) optimization formula:

PSV = w * profit + (1-w) * SLAV

Workload Statistics:
- Mean CPU Utilization: 65.5%
- Standard Deviation: 15.2%
- Min Utilization: 20.0%
- Max Utilization: 95.0%
- Violations Above Threshold (80%): 120
- Violations Below Minimum (10%): 50

Calculated Metrics:
- Profit: 75.30
- SLAV: 45.20

Determine the optimal weight w that maximizes PSV."""
    
    try:
        pipe = pipeline(
            task="text-generation",
            model=model,
            processing_class=tokenizer,
            max_length=500,
            temperature=0.3,
        )
        result = pipe(f"[INST] {prompt} [/INST]")
        print("\nTest Output:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"\nNote: Could not run test inference: {e}")
        print("Model training completed successfully. You can load and test the model separately.")


if __name__ == "__main__":
    main()

