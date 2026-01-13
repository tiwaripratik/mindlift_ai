# -*- coding: utf-8 -*-
"""
Test script for PlanetLab PSV Weight Optimization Model with Visualization

This script:
1. Loads 10-20 examples from the PlanetLab dataset
2. Tests the fine-tuned model on each example
3. Generates comprehensive plots showing before/after results

Usage (Local):
    python test_planetlab_with_plots.py

Usage (Google Colab):
    # Step 1: Mount Google Drive (if model is on Drive)
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Step 2: Clone or upload dataset
    !git clone https://github.com/beloglazov/planetlab-workload-traces.git
    # OR upload planetlab-workload-traces-master folder
    
    # Step 3: Navigate to your model directory (if needed)
    %cd /content/drive/MyDrive/path/to/model
    
    # Step 4: Run the script
    !python test_planetlab_with_plots.py
    
    # Step 5: Download plots (optional)
    from google.colab import files
    !zip -r test_plots.zip test_plots/
    files.download('test_plots.zip')
"""

import os
import sys
import json
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import re

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    # Enable inline plotting in Colab
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        pass  # Will use plt.show() instead
except ImportError:
    IN_COLAB = False

# Install required packages if in Colab
if IN_COLAB:
    print("Running in Google Colab environment")
    print("=" * 80)
    print("COLAB SETUP INSTRUCTIONS:")
    print("=" * 80)
    print("1. Mount Google Drive (if model is on Drive):")
    print("   from google.colab import drive")
    print("   drive.mount('/content/drive')")
    print("\n2. Upload dataset or clone repository:")
    print("   !git clone https://github.com/beloglazov/planetlab-workload-traces.git")
    print("   OR upload planetlab-workload-traces-master folder")
    print("\n3. Navigate to your model directory if needed:")
    print("   %cd /content/drive/MyDrive/path/to/model")
    print("=" * 80)
    
    required_packages = [
        ('transformers', '>=4.35.0'),
        ('peft', '>=0.7.0'),
        ('torch', '>=2.0.0'),
        ('numpy', '>=1.24.0'),
        ('matplotlib', '>=3.5.0'),
    ]
    
    import subprocess
    for package_name, version in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {package_name}{version}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    f"{package_name}{version}", "-q", "--upgrade"
                ])
            except Exception as e:
                print(f"Warning: Could not install {package_name}: {e}")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
)
from peft import PeftModel

# Suppress warnings
logging.set_verbosity(logging.CRITICAL)

# Configuration (should match training script)
PROFIT_PER_UNIT_UTILIZATION = 1.0
SLA_THRESHOLD = 80.0
SLA_PENALTY_PER_VIOLATION = 10.0
MIN_UTILIZATION_FOR_PROFIT = 10.0

# Number of examples to test
NUM_EXAMPLES = 15


def load_planetlab_traces(dataset_dir: str, max_traces: int = NUM_EXAMPLES) -> List[Dict]:
    """
    Load PlanetLab workload traces from directory structure.
    
    Args:
        dataset_dir: Path to planetlab-workload-traces-master directory
        max_traces: Maximum number of traces to load
        
    Returns:
        List of dictionaries with VM trace data
    """
    traces = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all date directories (format: YYYYMMDD)
    date_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit() and len(d) == 8]
    
    if not date_dirs:
        raise ValueError(f"No date directories found in {dataset_dir}")
    
    print(f"Found {len(date_dirs)} date directories")
    
    # Iterate through date directories
    for date_dir in sorted(date_dirs):
        if len(traces) >= max_traces:
            break
            
        date_path = os.path.join(dataset_dir, date_dir)
        
        # Get all files in this directory (they have no extension)
        all_files = [f for f in os.listdir(date_path) 
                     if os.path.isfile(os.path.join(date_path, f))]
        
        for trace_file in all_files:
            if len(traces) >= max_traces:
                break
                
            trace_path = os.path.join(date_path, trace_file)
            
            try:
                # Read CPU utilization values (one per line)
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
                            # Normalize to 0-100 range
                            if value > 100:
                                value = 100.0
                            elif value < 0:
                                value = 0.0
                            cpu_values.append(value)
                        except ValueError:
                            continue
                
                # Skip files with too few data points
                if len(cpu_values) < 50:
                    continue
                
                # Limit to first 500 points for visualization
                cpu_values = cpu_values[:500]
                
                # Store trace data
                traces.append({
                    'date': date_dir,
                    'vm_id': os.path.basename(trace_file),
                    'cpu_utilization': np.array(cpu_values),
                    'trace_file': trace_path
                })
                
            except Exception as e:
                print(f"Warning: Could not read {trace_file}: {e}")
                continue
    
    print(f"Loaded {len(traces)} VM traces")
    return traces


def calculate_profit(cpu_utilization: np.ndarray) -> float:
    """Calculate profit from CPU utilization"""
    if len(cpu_utilization) == 0:
        return 0.0
    
    avg_utilization = np.mean(cpu_utilization)
    base_profit = avg_utilization * PROFIT_PER_UNIT_UTILIZATION
    
    utilization_std = np.std(cpu_utilization)
    stability_bonus = max(0, (100 - utilization_std) / 100) * 10
    
    if avg_utilization > 90:
        overload_penalty = (avg_utilization - 90) * 2
    else:
        overload_penalty = 0
    
    profit = base_profit + stability_bonus - overload_penalty
    return max(0.0, profit)


def calculate_slav(cpu_utilization: np.ndarray) -> float:
    """Calculate SLA Violations"""
    if len(cpu_utilization) == 0:
        return 100.0
    
    violations_above = np.sum(cpu_utilization > SLA_THRESHOLD)
    violations_below = np.sum(cpu_utilization < MIN_UTILIZATION_FOR_PROFIT)
    total_violations = violations_above + violations_below
    
    violation_rate = total_violations / len(cpu_utilization)
    slav = violation_rate * SLA_PENALTY_PER_VIOLATION * 100
    
    if violations_above > 0:
        avg_excess = np.mean(cpu_utilization[cpu_utilization > SLA_THRESHOLD] - SLA_THRESHOLD)
        slav += avg_excess * 0.5
    
    return slav


def extract_workload_features(cpu_utilization: np.ndarray) -> Dict:
    """Extract features from workload trace"""
    if len(cpu_utilization) == 0:
        return {}
    
    return {
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


def create_prompt(features: Dict, profit: float, slav: float) -> str:
    """Create input prompt for the model"""
    prompt = f"""Analyze the following VM workload trace and determine the optimal weight parameter (w) 
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
    
    return prompt


def format_prompt_for_model(prompt: str) -> str:
    """Format prompt using the same format as training"""
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def extract_weight_from_response(response: str) -> Optional[float]:
    """Extract weight w from model response"""
    response_lower = response.lower()
    
    # Priority 1: Look for explicit weight declarations
    explicit_patterns = [
        r'optimal\s+weight\s+parameter\s*\(?\s*w\s*\)?\s*:\s*([0-9]\.[0-9]+)',
        r'optimal\s+weight\s+w\s*=\s*([0-9]\.[0-9]+)',
        r'weight\s+parameter\s*\(?\s*w\s*\)?\s*:\s*([0-9]\.[0-9]+)',
        r'optimal\s+w\s*=\s*([0-9]\.[0-9]+)',
        r'w\s*=\s*([0-9]\.[0-9]+)',
        r'weight\s*:\s*([0-9]\.[0-9]+)',
    ]
    
    for pattern in explicit_patterns:
        matches = re.finditer(pattern, response_lower, re.IGNORECASE)
        for match in matches:
            try:
                weight = float(match.group(1))
                if 0.0 <= weight <= 1.0:
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(response_lower), match.end() + 30)
                    context = response_lower[context_start:context_end]
                    
                    if 'coefficient' in context or 'variation' in context:
                        continue
                    if 'profit' in context and 'metric' in context:
                        continue
                    
                    return weight
            except ValueError:
                continue
    
    # Priority 2: Look for numbers that are clearly weights (0.x format)
    weight_pattern = r'\b(0\.[0-9]{4,})\b'
    matches = re.finditer(weight_pattern, response)
    for match in matches:
        try:
            weight = float(match.group(1))
            context_start = max(0, match.start() - 50)
            context_end = min(len(response_lower), match.end() + 50)
            context = response_lower[context_start:context_end]
            
            skip_indicators = ['coefficient', 'variation', 'profit', 'slav', 'utilization', 'threshold']
            if any(indicator in context for indicator in skip_indicators):
                if 'weight' not in context and 'w' not in context:
                    continue
            
            if 'weight' in context or 'w =' in context or 'optimal' in context:
                return weight
        except ValueError:
            continue
    
    # Priority 3: Find any number between 0 and 1
    all_numbers = re.findall(r'\b([0-9]\.[0-9]{2,})\b', response)
    for num_str in all_numbers:
        try:
            num = float(num_str)
            if 0.0 < num <= 1.0:
                return num
        except ValueError:
            pass
    
    return None


def test_model_on_traces(model, tokenizer, traces: List[Dict]) -> List[Dict]:
    """Test the model on PlanetLab traces"""
    results = []
    
    print("\n" + "=" * 80)
    print("Testing Model on PlanetLab Dataset")
    print("=" * 80 + "\n")
    
    for i, trace in enumerate(traces, 1):
        print(f"\n{'='*80}")
        print(f"Test Example {i}/{len(traces)}: {trace['vm_id']}")
        print(f"Date: {trace['date']}")
        print(f"{'='*80}")
        
        cpu_util = trace['cpu_utilization']
        features = extract_workload_features(cpu_util)
        profit = calculate_profit(cpu_util)
        slav = calculate_slav(cpu_util)
        
        print(f"\nWorkload Statistics:")
        print(f"  Mean CPU Utilization: {features['mean_utilization']:.2f}%")
        print(f"  Standard Deviation: {features['std_utilization']:.2f}%")
        print(f"  Min/Max: {features['min_utilization']:.2f}% / {features['max_utilization']:.2f}%")
        print(f"  Violations Above {SLA_THRESHOLD}%: {features['violations_above_threshold']}")
        print(f"  Violations Below {MIN_UTILIZATION_FOR_PROFIT}%: {features['violations_below_minimum']}")
        print(f"\nCalculated Metrics:")
        print(f"  Profit: {profit:.4f}")
        print(f"  SLAV: {slav:.4f}")
        
        # Create prompt
        prompt = create_prompt(features, profit, slav)
        formatted_prompt = format_prompt_for_model(prompt)
        
        # Generate response
        print(f"\nQuerying model...")
        try:
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            if "### Response:" in generated_text:
                generated_text = generated_text.split("### Response:")[-1].strip()
            
            # Extract weight
            predicted_w = extract_weight_from_response(generated_text)
            
            if predicted_w is not None:
                print(f"✓ Predicted Weight: w = {predicted_w:.6f}")
                
                # Calculate what optimal weight should be based on profit/SLAV ratio
                if slav > 0:
                    profit_slav_ratio = profit / slav
                    # When SLAV is high relative to profit, weight should be lower
                    # When profit is high relative to SLAV, weight should be higher
                    if profit_slav_ratio > 1.0:
                        expected_weight_range = "High (0.6-0.9)"  # Profit is more important
                    elif profit_slav_ratio > 0.5:
                        expected_weight_range = "Medium (0.4-0.7)"  # Balanced
                    elif profit_slav_ratio > 0.1:
                        expected_weight_range = "Low-Medium (0.2-0.5)"  # SLAV is more important
                    else:
                        expected_weight_range = "Low (0.1-0.3)"  # SLAV is much more important
                    
                    print(f"  Profit/SLAV Ratio: {profit_slav_ratio:.4f}")
                    print(f"  Expected Weight Range: {expected_weight_range}")
                    
                    # Analyze if prediction makes sense
                    if slav > 500 and predicted_w > 0.8:
                        print(f"  ⚠ WARNING: SLAV is very high ({slav:.2f}) but weight is high ({predicted_w:.4f})")
                        print(f"     → Model should prioritize SLAV minimization (lower weight)")
                    elif profit > 40 and slav < 50 and predicted_w < 0.5:
                        print(f"  ⚠ WARNING: Profit is high ({profit:.2f}) and SLAV is low ({slav:.2f}) but weight is low ({predicted_w:.4f})")
                        print(f"     → Model should prioritize profit maximization (higher weight)")
                    elif 0.3 <= predicted_w <= 0.7:
                        print(f"  ✓ Weight is balanced - considering both profit and SLAV")
                    elif predicted_w > 0.9:
                        print(f"  ⚠ Weight is very high - heavily prioritizing profit, ignoring SLAV")
                    elif predicted_w < 0.2:
                        print(f"  ⚠ Weight is very low - heavily prioritizing SLAV, ignoring profit")
                
                # Calculate PSV with predicted weight
                psv_predicted = predicted_w * profit + (1 - predicted_w) * slav
                
                # Calculate what PSV would be with different weights for comparison
                if slav > 0:
                    # Try a lower weight if SLAV is high
                    if slav > profit * 2:
                        alt_weight = 0.3  # Prioritize SLAV
                        psv_alt = alt_weight * profit + (1 - alt_weight) * slav
                        print(f"  PSV (predicted w={predicted_w:.4f}) = {psv_predicted:.4f}")
                        print(f"  PSV (if w=0.3, prioritizing SLAV) = {psv_alt:.4f}")
                        if psv_alt < psv_predicted:
                            print(f"  → Lower weight would give better PSV! (SLAV is too high)")
                    else:
                        print(f"  PSV (predicted) = {psv_predicted:.4f}")
                else:
                    print(f"  PSV (predicted) = {psv_predicted:.4f}")
                
                # Show model response snippet for debugging
                response_preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                print(f"\n  Model Response Preview: {response_preview}")
                
                # Calculate profit/SLAV ratio for analysis
                profit_slav_ratio = profit / slav if slav > 0 else float('inf')
                
                results.append({
                    'test_id': i,
                    'vm_id': trace['vm_id'],
                    'date': trace['date'],
                    'cpu_utilization': cpu_util,
                    'features': features,
                    'profit': profit,
                    'slav': slav,
                    'profit_slav_ratio': profit_slav_ratio,
                    'predicted_w': predicted_w,
                    'psv_predicted': psv_predicted,
                    'model_response': generated_text,
                    'success': True,
                })
            else:
                print(f"⚠ Warning: Could not extract weight from response")
                results.append({
                    'test_id': i,
                    'vm_id': trace['vm_id'],
                    'date': trace['date'],
                    'cpu_utilization': cpu_util,
                    'features': features,
                    'profit': profit,
                    'slav': slav,
                    'predicted_w': None,
                    'psv_predicted': None,
                    'model_response': generated_text,
                    'success': False,
                })
                
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            results.append({
                'test_id': i,
                'vm_id': trace['vm_id'],
                'date': trace['date'],
                'cpu_utilization': cpu_util,
                'features': features,
                'profit': profit,
                'slav': slav,
                'predicted_w': None,
                'psv_predicted': None,
                'model_response': None,
                'error': str(e),
                'success': False,
            })
    
    return results


def plot_results(results: List[Dict], output_dir: str = "./test_plots"):
    """Create comprehensive plots showing before/after results"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # In Colab, display plots inline
    if IN_COLAB:
        plt.ion()  # Turn on interactive mode for Colab
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("\n⚠ No successful results to plot")
        return
    
    print(f"\n{'='*80}")
    print(f"Generating Plots for {len(successful_results)} Examples")
    print(f"{'='*80}")
    
    # Set style (try different style names for compatibility)
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    # 1. Individual plots for each example
    print("\nCreating individual example plots...")
    for i, result in enumerate(successful_results):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Example {result['test_id']}: {result['vm_id']}\nDate: {result['date']}", 
                     fontsize=16, fontweight='bold')
        
        cpu_util = result['cpu_utilization']
        time_points = np.arange(len(cpu_util))
        
        # Plot 1: CPU Utilization Over Time (BEFORE)
        ax1 = axes[0, 0]
        ax1.plot(time_points, cpu_util, 'b-', linewidth=1.5, alpha=0.7, label='CPU Utilization')
        ax1.axhline(y=SLA_THRESHOLD, color='r', linestyle='--', linewidth=2, label=f'SLA Threshold ({SLA_THRESHOLD}%)')
        ax1.axhline(y=MIN_UTILIZATION_FOR_PROFIT, color='orange', linestyle='--', linewidth=2, 
                   label=f'Min Profit ({MIN_UTILIZATION_FOR_PROFIT}%)')
        ax1.fill_between(time_points, 0, cpu_util, alpha=0.3, color='blue')
        ax1.set_xlabel('Time (samples)', fontsize=12)
        ax1.set_ylabel('CPU Utilization (%)', fontsize=12)
        ax1.set_title('BEFORE: CPU Utilization Trace', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Metrics Comparison (BEFORE)
        ax2 = axes[0, 1]
        metrics = ['Profit', 'SLAV']
        values = [result['profit'], result['slav']]
        colors = ['green', 'red']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('BEFORE: Profit and SLAV Metrics', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 3: Model Prediction (AFTER)
        ax3 = axes[1, 0]
        predicted_w = result['predicted_w']
        psv_predicted = result['psv_predicted']
        
        # Show weight distribution
        weights = ['Profit Weight (w)', 'SLAV Weight (1-w)']
        weight_values = [predicted_w, 1 - predicted_w]
        colors_weights = ['green', 'red']
        bars = ax3.bar(weights, weight_values, color=colors_weights, alpha=0.7, edgecolor='black', linewidth=2)
        ax3.set_ylabel('Weight Value', fontsize=12)
        ax3.set_title(f'AFTER: Model Predicted Weights\nPSV = {psv_predicted:.4f}', 
                      fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, weight_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 4: PSV Components Breakdown
        ax4 = axes[1, 1]
        components = ['w × Profit', '(1-w) × SLAV', 'PSV']
        component_values = [
            predicted_w * result['profit'],
            (1 - predicted_w) * result['slav'],
            psv_predicted
        ]
        colors_comp = ['green', 'red', 'blue']
        bars = ax4.bar(components, component_values, color=colors_comp, alpha=0.7, 
                      edgecolor='black', linewidth=2)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title('AFTER: PSV Components Breakdown', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, component_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = os.path.join(output_dir, f"example_{result['test_id']:02d}_{result['vm_id'][:30]}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        
        # Display in Colab
        if IN_COLAB:
            plt.show()
        
        plt.close()
        
        print(f"  ✓ Saved: {plot_filename}")
    
    # 2. Summary comparison plot
    print("\nCreating summary comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Summary Across All Examples', fontsize=16, fontweight='bold')
    
    # Extract data for summary
    test_ids = [r['test_id'] for r in successful_results]
    profits = [r['profit'] for r in successful_results]
    slavs = [r['slav'] for r in successful_results]
    predicted_ws = [r['predicted_w'] for r in successful_results]
    psvs = [r['psv_predicted'] for r in successful_results]
    
    # Plot 1: Profit vs SLAV scatter
    ax1 = axes[0, 0]
    scatter = ax1.scatter(profits, slavs, c=predicted_ws, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Profit', fontsize=12)
    ax1.set_ylabel('SLAV', fontsize=12)
    ax1.set_title('Profit vs SLAV (colored by predicted weight)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Predicted Weight (w)')
    
    # Plot 2: Predicted weights distribution
    ax2 = axes[0, 1]
    ax2.hist(predicted_ws, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(predicted_ws), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(predicted_ws):.4f}')
    ax2.set_xlabel('Predicted Weight (w)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Predicted Weights', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: PSV values
    ax3 = axes[1, 0]
    ax3.bar(range(len(psvs)), psvs, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Example ID', fontsize=12)
    ax3.set_ylabel('PSV Value', fontsize=12)
    ax3.set_title('Predicted PSV Values Across Examples', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(test_ids)))
    ax3.set_xticklabels(test_ids, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Weight vs Profit/SLAV ratio with expected behavior
    ax4 = axes[1, 1]
    profit_slav_ratios = [r.get('profit_slav_ratio', p/s if s > 0 else 0) for r, p, s in zip(successful_results, profits, slavs)]
    ax4.scatter(profit_slav_ratios, predicted_ws, s=100, alpha=0.7, 
               c=psvs, cmap='coolwarm', edgecolors='black', linewidth=1.5)
    
    # Add expected weight curve (heuristic: higher ratio -> higher weight, but with diminishing returns)
    if len(profit_slav_ratios) > 0:
        sorted_ratios = sorted([r for r in profit_slav_ratios if r > 0])
        if sorted_ratios:
            x_curve = np.linspace(min(sorted_ratios), max(sorted_ratios), 100)
            # Expected weight: sigmoid-like curve, but capped
            # When ratio is very low (<0.1), weight should be low (0.1-0.3)
            # When ratio is medium (0.1-1.0), weight should be medium (0.3-0.7)
            # When ratio is high (>1.0), weight should be high (0.7-0.9)
            y_expected = np.clip(0.1 + 0.8 * (1 / (1 + np.exp(-5 * (x_curve - 0.5)))), 0.1, 0.9)
            ax4.plot(x_curve, y_expected, 'r--', linewidth=2, label='Expected Weight Trend', alpha=0.7)
    
    ax4.set_xlabel('Profit/SLAV Ratio', fontsize=12)
    ax4.set_ylabel('Predicted Weight (w)', fontsize=12)
    ax4.set_title('Weight vs Profit/SLAV Ratio\n(Red line: Expected trend)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.colorbar(ax4.collections[0], ax=ax4, label='PSV Value')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = os.path.join(output_dir, "summary_comparison.png")
    plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
    
    # Display in Colab
    if IN_COLAB:
        plt.show()
    
    plt.close()
    
    print(f"  ✓ Saved: {summary_filename}")
    
    # 3. Before/After comparison for first few examples
    print("\nCreating before/after comparison plots...")
    num_comparison = min(5, len(successful_results))
    
    for i in range(num_comparison):
        result = successful_results[i]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Example {result["test_id"]}: {result["vm_id"]} - Before vs After', 
                     fontsize=16, fontweight='bold')
        
        cpu_util = result['cpu_utilization']
        time_points = np.arange(len(cpu_util))
        
        # BEFORE plot
        ax1 = axes[0]
        ax1.plot(time_points, cpu_util, 'b-', linewidth=2, alpha=0.7)
        ax1.axhline(y=SLA_THRESHOLD, color='r', linestyle='--', linewidth=2, 
                   label=f'SLA Threshold ({SLA_THRESHOLD}%)')
        ax1.axhline(y=MIN_UTILIZATION_FOR_PROFIT, color='orange', linestyle='--', linewidth=2,
                   label=f'Min Profit ({MIN_UTILIZATION_FOR_PROFIT}%)')
        ax1.fill_between(time_points, 0, cpu_util, alpha=0.3, color='blue')
        ax1.set_xlabel('Time (samples)', fontsize=12)
        ax1.set_ylabel('CPU Utilization (%)', fontsize=12)
        ax1.set_title(f'BEFORE\nProfit: {result["profit"]:.2f}, SLAV: {result["slav"]:.2f}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # AFTER plot (showing optimization result)
        ax2 = axes[1]
        # Show the same CPU trace but with annotations
        ax2.plot(time_points, cpu_util, 'b-', linewidth=2, alpha=0.7)
        ax2.axhline(y=SLA_THRESHOLD, color='r', linestyle='--', linewidth=2, 
                   label=f'SLA Threshold ({SLA_THRESHOLD}%)')
        ax2.axhline(y=MIN_UTILIZATION_FOR_PROFIT, color='orange', linestyle='--', linewidth=2,
                   label=f'Min Profit ({MIN_UTILIZATION_FOR_PROFIT}%)')
        ax2.fill_between(time_points, 0, cpu_util, alpha=0.3, color='blue')
        
        # Add text box with model predictions
        textstr = f'Model Prediction:\nWeight (w): {result["predicted_w"]:.4f}\nPSV: {result["psv_predicted"]:.2f}\n\nOptimization:\nProfit Weight: {result["predicted_w"]:.2%}\nSLAV Weight: {1-result["predicted_w"]:.2%}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        ax2.set_xlabel('Time (samples)', fontsize=12)
        ax2.set_ylabel('CPU Utilization (%)', fontsize=12)
        ax2.set_title(f'AFTER\nOptimized PSV: {result["psv_predicted"]:.2f}', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_filename = os.path.join(output_dir, f"before_after_{result['test_id']:02d}.png")
        plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
        
        # Display in Colab
        if IN_COLAB:
            plt.show()
        
        plt.close()
        
        print(f"  ✓ Saved: {comparison_filename}")
    
    print(f"\n{'='*80}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*80}")
    
    # In Colab, provide download instructions
    if IN_COLAB:
        print("\n💡 To download plots in Colab, run:")
        print("   from google.colab import files")
        print("   import zipfile")
        print("   import os")
        print("   ")
        print("   # Create zip file")
        print("   with zipfile.ZipFile('test_plots.zip', 'w') as zipf:")
        print("       for root, dirs, files in os.walk('test_plots'):")
        print("           for file in files:")
        print("               zipf.write(os.path.join(root, file))")
        print("   ")
        print("   # Download")
        print("   files.download('test_plots.zip')")


def main():
    """Main test function"""
    print("=" * 80)
    print("PlanetLab PSV Weight Optimization Model - Test Script with Visualization")
    print("=" * 80)
    
    # Check if model exists
    model_dir = None
    possible_model_dirs = [
        "./planetlab",
        "./final_model",
        "planetlab",
        "final_model",
    ]
    
    # Add Colab-specific paths
    if IN_COLAB:
        possible_model_dirs.extend([
            "/content/drive/MyDrive/hh/planetlab",  # User's specific path
            "/content/drive/MyDrive/planetlab",
            "/content/drive/MyDrive/final_model",
            "/content/planetlab",
            "/content/final_model",
        ])
    
    for possible_dir in possible_model_dirs:
        if os.path.exists(possible_dir):
            if os.path.exists(os.path.join(possible_dir, "adapter_config.json")) or \
               os.path.exists(os.path.join(possible_dir, "adapter_model.safetensors")):
                model_dir = possible_dir
                print(f"\n✓ Found model directory: {model_dir}")
                break
    
    if model_dir is None:
        print(f"\n✗ ERROR: Model directory not found")
        print("Please ensure the model is in ./planetlab directory")
        return
    
    # Load model
    print(f"\n1. Loading model from {model_dir}...")
    try:
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            print("   ✓ Loaded tokenizer from model directory")
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            print("   ✓ Loaded tokenizer from base model")
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load base model
        print("   Loading base model...")
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("   ✓ Base model loaded")
        
        # Load PEFT adapter
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            print("   Loading PEFT adapter...")
            model = PeftModel.from_pretrained(model, model_dir)
            print("   ✓ PEFT adapter loaded")
        
        model.eval()
        print("   ✓ Model loaded successfully")
        
        if torch.cuda.is_available():
            print(f"   Using device: CUDA (GPU)")
        else:
            print(f"   Using device: CPU")
            
    except Exception as e:
        print(f"   ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load PlanetLab traces
    print(f"\n2. Loading PlanetLab traces...")
    
    # Try multiple possible dataset locations
    possible_dataset_dirs = [
        "./planetlab-workload-traces-master",
        "./planetlab-workload-traces",
        "planetlab-workload-traces-master",
        "planetlab-workload-traces",
    ]
    
    if IN_COLAB:
        possible_dataset_dirs.extend([
            "/content/drive/MyDrive/hh/planetlab-workload-traces-master",  # User's specific path
            "/content/planetlab-workload-traces-master",
            "/content/planetlab-workload-traces",
            "/content/drive/MyDrive/planetlab-workload-traces-master",
            "/content/drive/MyDrive/planetlab-workload-traces",
        ])
    
    dataset_dir = None
    for possible_dir in possible_dataset_dirs:
        if os.path.exists(possible_dir):
            dataset_dir = possible_dir
            print(f"   ✓ Found dataset at: {dataset_dir}")
            break
    
    if dataset_dir is None:
        print(f"   ✗ ERROR: Dataset directory not found in any of these locations:")
        for possible_dir in possible_dataset_dirs:
            print(f"      - {possible_dir}")
        print("\n   Please ensure the dataset is available.")
        if IN_COLAB:
            print("   You can clone it with:")
            print("   !git clone https://github.com/beloglazov/planetlab-workload-traces.git")
        return
    
    try:
        traces = load_planetlab_traces(dataset_dir, max_traces=NUM_EXAMPLES)
        print(f"   ✓ Loaded {len(traces)} traces")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to load traces: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model
    print(f"\n3. Testing model on {len(traces)} examples...")
    results = test_model_on_traces(model, tokenizer, traces)
    
    # Generate plots
    print(f"\n4. Generating plots...")
    plot_results(results)
    
    # Save results
    output_file = "./planetlab_test_results_with_plots.json"
    print(f"\n5. Saving results to {output_file}...")
    try:
        # Convert numpy types to Python native types for JSON
        json_results = []
        for r in results:
            json_result = {}
            for key, value in r.items():
                if key == 'cpu_utilization':
                    json_result[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result[key] = float(value)
                elif isinstance(value, np.generic):
                    json_result[key] = value.item()
                elif isinstance(value, dict):
                    json_result[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating, np.generic)):
                            json_result[key][k] = float(v) if isinstance(v, (np.floating, np.generic)) else int(v)
                        elif isinstance(v, np.ndarray):
                            json_result[key][k] = v.tolist()
                        else:
                            json_result[key][k] = v
                else:
                    json_result[key] = value
            json_results.append(json_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Results saved successfully")
        
    except Exception as e:
        print(f"   ⚠ Warning: Could not save results: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        weights = [r['predicted_w'] for r in successful_tests]
        profits = [r['profit'] for r in successful_tests]
        slavs = [r['slav'] for r in successful_tests]
        profit_slav_ratios = [r.get('profit_slav_ratio', p/s if s > 0 else 0) for r, p, s in zip(successful_tests, profits, slavs)]
        
        print(f"\nModel Performance:")
        print(f"  Average Weight: {np.mean(weights):.6f}")
        print(f"  Std Dev Weight: {np.std(weights):.6f}")
        print(f"  Min Weight: {min(weights):.6f}")
        print(f"  Max Weight: {max(weights):.6f}")
        
        # Check if weights are too uniform (potential issue)
        if np.std(weights) < 0.01:
            print(f"\n  ⚠ WARNING: Weights are very uniform (std={np.std(weights):.6f})")
            print(f"     → Model may not be adapting to different profit/SLAV scenarios")
            print(f"     → Expected: Weights should vary based on profit/SLAV ratio")
        
        avg_profit = np.mean(profits)
        avg_slav = np.mean(slavs)
        avg_psv = np.mean([r['psv_predicted'] for r in successful_tests])
        avg_ratio = np.mean([r for r in profit_slav_ratios if r > 0])
        
        print(f"\nAverage Metrics:")
        print(f"  Profit: {avg_profit:.4f}")
        print(f"  SLAV: {avg_slav:.4f}")
        print(f"  Profit/SLAV Ratio: {avg_ratio:.4f}")
        print(f"  PSV: {avg_psv:.4f}")
        
        # Analyze weight distribution
        high_slav_cases = [r for r in successful_tests if r['slav'] > 500]
        low_slav_cases = [r for r in successful_tests if r['slav'] < 100]
        
        if high_slav_cases:
            high_slav_weights = [r['predicted_w'] for r in high_slav_cases]
            print(f"\nHigh SLAV Cases (SLAV > 500): {len(high_slav_cases)}")
            print(f"  Average Weight: {np.mean(high_slav_weights):.4f}")
            print(f"  Expected: Lower weights (0.1-0.4) to prioritize SLAV minimization")
            if np.mean(high_slav_weights) > 0.7:
                print(f"  ⚠ PROBLEM: Weights are too high for high SLAV cases!")
                print(f"     → Model should predict lower weights when SLAV is high")
        
        if low_slav_cases:
            low_slav_weights = [r['predicted_w'] for r in low_slav_cases]
            print(f"\nLow SLAV Cases (SLAV < 100): {len(low_slav_cases)}")
            print(f"  Average Weight: {np.mean(low_slav_weights):.4f}")
            print(f"  Expected: Higher weights (0.6-0.9) to prioritize profit maximization")
            if np.mean(low_slav_weights) < 0.5:
                print(f"  ⚠ PROBLEM: Weights are too low for low SLAV cases!")
                print(f"     → Model should predict higher weights when SLAV is low")
    
    print("\n" + "=" * 80)
    print("Testing completed! Check ./test_plots/ for visualization results.")
    print("=" * 80)
    
    if IN_COLAB:
        print("\n📊 Plots are displayed above and saved to ./test_plots/")
        print("💾 To download all plots, see instructions above or run:")
        print("   from google.colab import files")
        print("   !zip -r test_plots.zip test_plots/")
        print("   files.download('test_plots.zip')")


if __name__ == "__main__":
    main()

