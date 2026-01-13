# -*- coding: utf-8 -*-
"""
Test script for fine-tuned Mistral 7B PSV Weight Optimization Model
COLAB-COMPATIBLE WITH GOOGLE DRIVE SUPPORT

This script loads the fine-tuned model from ./planetlab folder and tests it with
various VM workload scenarios to predict optimal weight parameter (w) for
Profit-SLA Violation (PSV) optimization.

Usage:
    # In Colab with Google Drive:
    # 1. Mount Google Drive:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # 2. Navigate to your model folder:
    %cd /content/drive/MyDrive/path/to/your/model
    
    # 3. Run the test:
    !python test_planetlab_model.py
    
    # Or locally:
    python test_planetlab_model.py

Requirements:
    - The model must be trained and saved to ./planetlab directory
    - Required packages: transformers, peft, torch, numpy

What it tests:
    1. Optimal weight prediction for individual VM workloads
    2. Optimal weight prediction for aggregated workloads
    3. PSV calculation with predicted weights
    4. Model performance on various workload scenarios
    5. SLAV minimization verification
    6. Profit maximization verification

Output:
    - Prints test results to console
    - Saves detailed results to ./planetlab_test_results.json
"""

import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Optional

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install required packages if in Colab
if IN_COLAB:
    print("Running in Google Colab environment")
    print("Checking dependencies...")
    
    required_packages = [
        ('transformers', '>=4.35.0'),
        ('peft', '>=0.7.0'),
        ('torch', '>=2.0.0'),
        ('numpy', '>=1.24.0'),
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


def calculate_profit(cpu_utilization: np.ndarray) -> float:
    """Calculate profit from CPU utilization (same as training script)"""
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
    """Calculate SLA Violations (same as training script)"""
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


def create_test_workloads() -> List[Dict]:
    """Create test workload scenarios"""
    workloads = []
    
    # Test Case 1: High utilization, stable workload
    cpu_high_stable = np.random.normal(75, 5, 100)  # Mean 75%, std 5%
    cpu_high_stable = np.clip(cpu_high_stable, 0, 100)
    workloads.append({
        'name': 'High Utilization, Stable',
        'cpu_utilization': cpu_high_stable,
        'description': 'High CPU utilization with low variance (good for profit)'
    })
    
    # Test Case 2: Medium utilization, some violations
    cpu_medium = np.random.normal(60, 20, 100)  # Mean 60%, std 20%
    cpu_medium = np.clip(cpu_medium, 0, 100)
    workloads.append({
        'name': 'Medium Utilization, Variable',
        'cpu_utilization': cpu_medium,
        'description': 'Moderate utilization with some variability'
    })
    
    # Test Case 3: Low utilization, many violations
    cpu_low = np.random.normal(30, 10, 100)  # Mean 30%, std 10%
    cpu_low = np.clip(cpu_low, 0, 100)
    workloads.append({
        'name': 'Low Utilization, Many SLA Violations',
        'cpu_utilization': cpu_low,
        'description': 'Low utilization leading to SLA violations (waste)'
    })
    
    # Test Case 4: Very high utilization, overload risk
    cpu_overload = np.random.normal(95, 3, 100)  # Mean 95%, std 3%
    cpu_overload = np.clip(cpu_overload, 0, 100)
    workloads.append({
        'name': 'Very High Utilization, Overload Risk',
        'cpu_utilization': cpu_overload,
        'description': 'Very high utilization with overload risk'
    })
    
    # Test Case 5: Optimal utilization
    cpu_optimal = np.random.normal(70, 8, 100)  # Mean 70%, std 8%
    cpu_optimal = np.clip(cpu_optimal, 0, 100)
    workloads.append({
        'name': 'Optimal Utilization',
        'cpu_utilization': cpu_optimal,
        'description': 'Optimal utilization balance (70-80% range)'
    })
    
    # Test Case 6: Highly variable workload
    cpu_variable = np.concatenate([
        np.random.normal(90, 5, 30),
        np.random.normal(40, 5, 30),
        np.random.normal(70, 5, 40)
    ])
    cpu_variable = np.clip(cpu_variable, 0, 100)
    workloads.append({
        'name': 'Highly Variable Workload',
        'cpu_utilization': cpu_variable,
        'description': 'Highly variable workload with spikes and dips'
    })
    
    return workloads


def extract_workload_features(cpu_utilization: np.ndarray) -> Dict:
    """Extract features from workload trace (same as training script)"""
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
    """Create input prompt for the model (matches training format)"""
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
    """Format prompt using the same format as training (### Instruction: / ### Response:)"""
    # Match the training format from finetuning3.py formatting_func
    return f"### Instruction:\n{prompt}\n\n### Response:\n"


def extract_weight_from_response(response: str) -> Optional[float]:
    """Extract weight w from model response"""
    import re
    
    response_lower = response.lower()
    
    # Priority 1: Look for explicit weight declarations with context
    explicit_patterns = [
        r'optimal\s+weight\s+parameter\s*\(?\s*w\s*\)?\s*:\s*([0-9]\.[0-9]+)',  # "Optimal Weight Parameter (w): 0.123"
        r'optimal\s+weight\s+w\s*=\s*([0-9]\.[0-9]+)',  # "optimal weight w = 0.123"
        r'weight\s+parameter\s*\(?\s*w\s*\)?\s*:\s*([0-9]\.[0-9]+)',  # "weight parameter (w): 0.123"
        r'optimal\s+w\s*=\s*([0-9]\.[0-9]+)',  # "optimal w = 0.123"
        r'w\s*=\s*([0-9]\.[0-9]+)',  # "w = 0.123"
        r'weight\s*:\s*([0-9]\.[0-9]+)',  # "weight: 0.123"
    ]
    
    for pattern in explicit_patterns:
        matches = re.finditer(pattern, response_lower, re.IGNORECASE)
        for match in matches:
            try:
                weight = float(match.group(1))
                # Ensure weight is in [0, 1] range
                if 0.0 <= weight <= 1.0:
                    # Check context - avoid matching coefficient of variation or other metrics
                    context_start = max(0, match.start() - 30)
                    context_end = min(len(response_lower), match.end() + 30)
                    context = response_lower[context_start:context_end]
                    
                    # Skip if it's clearly about coefficient of variation
                    if 'coefficient' in context or 'variation' in context:
                        continue
                    # Skip if it's about profit or SLAV values
                    if 'profit' in context and 'metric' in context:
                        continue
                    
                    return weight
            except ValueError:
                continue
    
    # Priority 2: Look for numbers that are clearly weights (0.x format with 4+ decimal places)
    # This pattern matches weights like 0.9425, 0.9263, etc.
    weight_pattern = r'\b(0\.[0-9]{4,})\b'
    matches = re.finditer(weight_pattern, response)
    for match in matches:
        try:
            weight = float(match.group(1))
            # Check context to avoid false positives
            context_start = max(0, match.start() - 50)
            context_end = min(len(response_lower), match.end() + 50)
            context = response_lower[context_start:context_end]
            
            # Skip if it's clearly not a weight
            skip_indicators = ['coefficient', 'variation', 'profit', 'slav', 'utilization', 'threshold']
            if any(indicator in context for indicator in skip_indicators):
                # Only skip if it's near these indicators AND not near weight-related words
                if 'weight' not in context and 'w' not in context:
                    continue
            
            # Prefer weights that appear near weight-related keywords
            if 'weight' in context or 'w =' in context or 'optimal' in context:
                return weight
        except ValueError:
            continue
    
    # Priority 3: Find any number between 0 and 1, but be more careful
    # Look for numbers close to 1.0 (like 0.999994) or typical weight ranges
    very_close_to_one = r'\b(0\.9[0-9]{4,}|0\.99[0-9]{3,}|0\.999[0-9]{2,})\b'
    matches = re.finditer(very_close_to_one, response)
    for match in matches:
        try:
            weight = float(match.group(1))
            if 0.9 <= weight <= 1.0:
                return weight
        except ValueError:
            continue
    
    # Priority 4: Last resort - find any number between 0 and 1
    all_numbers = re.findall(r'\b([0-9]\.[0-9]{2,})\b', response)
    for num_str in all_numbers:
        try:
            num = float(num_str)
            if 0.0 < num <= 1.0:
                # Extra validation: skip if it looks like a metric value (too large)
                if num > 1.0:
                    continue
                return num
        except ValueError:
            pass
    
    return None


def find_optimal_weight_ground_truth(profit: float, slav: float) -> float:
    """Find optimal weight using optimization (for comparison)"""
    # Normalize profit and SLAV to 0-1 range for fair comparison
    # We'll use a simple approach: if profit/slav ratio is high, prefer profit
    if slav == 0:
        return 1.0  # Focus on profit if no SLA violations
    
    profit_to_slav_ratio = profit / slav
    
    # Simple heuristic: higher ratio -> higher weight
    # But we want to ensure w is in [0, 1]
    if profit_to_slav_ratio > 2.0:
        return min(0.95, 0.5 + profit_to_slav_ratio * 0.1)
    elif profit_to_slav_ratio > 1.0:
        return min(0.85, 0.4 + profit_to_slav_ratio * 0.15)
    elif profit_to_slav_ratio > 0.5:
        return min(0.7, 0.3 + profit_to_slav_ratio * 0.2)
    else:
        return max(0.1, 0.2 + profit_to_slav_ratio * 0.2)


def test_model(model, tokenizer, workloads: List[Dict]) -> List[Dict]:
    """Test the model with various workloads"""
    results = []
    
    print("\n" + "=" * 80)
    print("Testing Model with Workload Scenarios")
    print("=" * 80 + "\n")
    
    # Use text-generation pipeline with the formatted prompt
    # Note: We'll format the prompt manually to match training format
    for i, workload in enumerate(workloads, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {workload['name']}")
        print(f"{'='*80}")
        print(f"Description: {workload['description']}")
        
        cpu_util = workload['cpu_utilization']
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
        print(f"  Profit: {profit:.4f} (higher is better)")
        print(f"  SLAV: {slav:.4f} (lower is better)")
        
        # Calculate ground truth optimal weight for comparison
        optimal_w_ground_truth = find_optimal_weight_ground_truth(profit, slav)
        
        # Create prompt
        prompt = create_prompt(features, profit, slav)
        
        # Format prompt using training format
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
            
            # Extract only the response part (after "### Response:")
            if "### Response:" in generated_text:
                generated_text = generated_text.split("### Response:")[-1].strip()
            
            print(f"\nModel Response:")
            print("-" * 80)
            print(generated_text)
            print("-" * 80)
            
            # Extract weight
            predicted_w = extract_weight_from_response(generated_text)
            
            if predicted_w is not None:
                print(f"\n✓ Extracted Weight: w = {predicted_w:.6f}")
                print(f"  Ground Truth Optimal w: {optimal_w_ground_truth:.6f}")
                print(f"  Difference: {abs(predicted_w - optimal_w_ground_truth):.6f}")
                
                # Validation: Check if weight seems reasonable
                # If profit is high and SLAV is low, weight should be high
                # If profit is low and SLAV is high, weight should be lower
                profit_to_slav_ratio = profit / slav if slav > 0 else float('inf')
                
                # Check if model is optimizing correctly
                is_profit_focused = predicted_w > 0.6 and profit > 50
                is_slav_focused = predicted_w < 0.4 and slav > 50
                is_balanced = 0.4 <= predicted_w <= 0.6
                
                print(f"\n  Profit-to-SLAV Ratio: {profit_to_slav_ratio:.4f}")
                if profit_to_slav_ratio > 1.0 and predicted_w < 0.5:
                    print(f"  ⚠ Warning: Weight seems low ({predicted_w:.4f}) for high profit-to-SLAV ratio ({profit_to_slav_ratio:.2f})")
                    print(f"     Expected: Higher weight to maximize profit")
                elif profit_to_slav_ratio < 0.5 and predicted_w > 0.8:
                    print(f"  ⚠ Warning: Weight seems high ({predicted_w:.4f}) for low profit-to-SLAV ratio ({profit_to_slav_ratio:.2f})")
                    print(f"     Expected: Lower weight to minimize SLA violations")
                else:
                    print(f"  ✓ Weight seems reasonable for profit/SLAV ratio")
                
                # Calculate PSV with predicted weight
                psv_predicted = predicted_w * profit + (1 - predicted_w) * slav
                psv_optimal = optimal_w_ground_truth * profit + (1 - optimal_w_ground_truth) * slav
                
                # Calculate PSV improvement
                psv_improvement = ((psv_predicted - psv_optimal) / psv_optimal * 100) if psv_optimal != 0 else 0
                
                print(f"\nPSV Calculation:")
                print(f"  PSV (predicted w={predicted_w:.6f}) = {predicted_w:.6f} * {profit:.4f} + (1 - {predicted_w:.6f}) * {slav:.4f}")
                print(f"  PSV (predicted) = {psv_predicted:.4f}")
                print(f"  PSV (optimal w={optimal_w_ground_truth:.6f}) = {psv_optimal:.4f}")
                print(f"  PSV Improvement: {psv_improvement:.2f}%")
                
                # Verify SLAV minimization and profit maximization
                print(f"\nOptimization Verification:")
                print(f"  Profit Maximization: {'✓' if is_profit_focused or (predicted_w > 0.5 and profit > 30) else '⚠'}")
                print(f"    - Profit weight: {predicted_w:.4f} ({predicted_w*100:.1f}%)")
                print(f"    - Profit value: {profit:.4f}")
                print(f"  SLAV Minimization: {'✓' if is_slav_focused or (predicted_w < 0.6 and slav < 100) else '⚠'}")
                print(f"    - SLAV weight: {1-predicted_w:.4f} ({(1-predicted_w)*100:.1f}%)")
                print(f"    - SLAV value: {slav:.4f}")
                
                if predicted_w > 0.7:
                    print(f"  → Model prioritizes profit maximization (w = {predicted_w:.4f})")
                elif predicted_w < 0.3:
                    print(f"  → Model prioritizes SLA violation minimization (w = {predicted_w:.4f})")
                else:
                    print(f"  → Model balances profit and SLA violations (w = {predicted_w:.4f})")
                
                results.append({
                    'test_case': i,
                    'name': workload['name'],
                    'description': workload['description'],
                    'features': features,
                    'profit': profit,
                    'slav': slav,
                    'predicted_w': predicted_w,
                    'optimal_w_ground_truth': optimal_w_ground_truth,
                    'psv_predicted': psv_predicted,
                    'psv_optimal': psv_optimal,
                    'psv_improvement_percent': psv_improvement,
                    'profit_to_slav_ratio': profit_to_slav_ratio,
                    'model_response': generated_text,
                    'success': True,
                    'profit_maximized': is_profit_focused or (predicted_w > 0.5 and profit > 30),
                    'slav_minimized': is_slav_focused or (predicted_w < 0.6 and slav < 100),
                })
            else:
                print(f"\n⚠ Warning: Could not extract weight from response")
                results.append({
                    'test_case': i,
                    'name': workload['name'],
                    'description': workload['description'],
                    'features': features,
                    'profit': profit,
                    'slav': slav,
                    'predicted_w': None,
                    'psv': None,
                    'model_response': generated_text,
                    'success': False
                })
                
        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            results.append({
                'test_case': i,
                'name': workload['name'],
                'description': workload['description'],
                'features': features,
                'profit': profit,
                'slav': slav,
                'predicted_w': None,
                'psv': None,
                'model_response': None,
                'error': str(e),
                'success': False
            })
    
    return results


def main():
    """Main test function"""
    print("=" * 80)
    print("PlanetLab PSV Weight Optimization Model - Test Script")
    print("Google Drive / Colab Compatible")
    print("=" * 80)
    
    # Check for Google Drive mount in Colab
    if IN_COLAB:
        print("\n📁 Checking for Google Drive...")
        drive_paths = [
            "/content/drive/MyDrive",
            "/content/drive",
        ]
        for drive_path in drive_paths:
            if os.path.exists(drive_path):
                print(f"   ✓ Google Drive mounted at: {drive_path}")
                break
    
    # Check if model exists - try multiple possible locations
    model_dir = None
    possible_model_dirs = [
        "./planetlab",
        "./final_model",
        "planetlab",
        "final_model",
        "/content/drive/MyDrive/planetlab",
        "/content/drive/MyDrive/final_model",
        "/content/planetlab",
        "/content/final_model",
    ]
    
    for possible_dir in possible_model_dirs:
        if os.path.exists(possible_dir):
            # Check if it contains adapter files
            if os.path.exists(os.path.join(possible_dir, "adapter_config.json")) or \
               os.path.exists(os.path.join(possible_dir, "adapter_model.safetensors")) or \
               os.path.exists(os.path.join(possible_dir, "config.json")):
                model_dir = possible_dir
                print(f"\n✓ Found model directory: {model_dir}")
                break
    
    if model_dir is None:
        print(f"\n✗ ERROR: Model directory not found in any of these locations:")
        for possible_dir in possible_model_dirs:
            print(f"   - {possible_dir}")
        print("\nPlease ensure:")
        print("1. The model is trained and saved to ./planetlab directory")
        print("2. Or specify the correct path to your model directory")
        print("3. In Colab, make sure Google Drive is mounted if model is on Drive")
        return
    
    print(f"\n1. Loading model from {model_dir}...")
    try:
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # Load tokenizer (try from model_dir first, then base model)
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
        
        # Check if this is a PEFT adapter or full model
        is_peft_adapter = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
        
        if is_peft_adapter:
            print("   Loading PEFT adapter...")
            # Load PEFT adapter
            model = PeftModel.from_pretrained(model, model_dir)
            # Don't merge - use adapter directly for inference
            # model = model.merge_and_unload()  # Uncomment if you want to merge
            print("   ✓ PEFT adapter loaded")
        else:
            print("   ✓ Using full model (no adapter)")
        
        # Set to eval mode
        model.eval()
        
        print("   ✓ Model loaded successfully")
        
        if torch.cuda.is_available():
            print(f"   Using device: CUDA (GPU)")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"   Allocated: {allocated:.2f} GB")
        else:
            print(f"   Using device: CPU")
            
    except Exception as e:
        print(f"   ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure the model was trained successfully")
        print("2. Check that ./planetlab directory contains model files")
        print("3. Verify you have enough GPU/CPU memory")
        print("4. In Colab, ensure Google Drive is mounted if model is on Drive")
        return
    
    # Create test workloads
    print("\n2. Creating test workloads...")
    workloads = create_test_workloads()
    print(f"   ✓ Created {len(workloads)} test scenarios")
    
    # Test model
    print("\n3. Testing model...")
    results = test_model(model, tokenizer, workloads)
    
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
        print(f"\nSuccessful Predictions:")
        print("-" * 80)
        for result in successful_tests:
            print(f"  Test {result['test_case']}: {result['name']}")
            print(f"    Predicted w: {result['predicted_w']:.6f}")
            print(f"    PSV (predicted): {result['psv_predicted']:.4f}")
            print(f"    PSV (optimal): {result['psv_optimal']:.4f}")
            print(f"    Profit: {result['profit']:.4f}, SLAV: {result['slav']:.4f}")
            print(f"    Profit Maximized: {'✓' if result.get('profit_maximized', False) else '✗'}")
            print(f"    SLAV Minimized: {'✓' if result.get('slav_minimized', False) else '✗'}")
            print()
        
        # Calculate statistics
        weights = [r['predicted_w'] for r in successful_tests]
        psv_improvements = [r.get('psv_improvement_percent', 0) for r in successful_tests]
        profit_maximized_count = sum(1 for r in successful_tests if r.get('profit_maximized', False))
        slav_minimized_count = sum(1 for r in successful_tests if r.get('slav_minimized', False))
        
        avg_weight = np.mean(weights)
        std_weight = np.std(weights)
        avg_psv_improvement = np.mean(psv_improvements) if psv_improvements else 0
        
        print(f"\nModel Performance Statistics:")
        print(f"  Average w: {avg_weight:.6f}")
        print(f"  Std Dev w: {std_weight:.6f}")
        print(f"  Min w: {min(weights):.6f}")
        print(f"  Max w: {max(weights):.6f}")
        print(f"  Average PSV Improvement: {avg_psv_improvement:.2f}%")
        print(f"\nOptimization Verification:")
        print(f"  Profit Maximization: {profit_maximized_count}/{len(successful_tests)} tests ({profit_maximized_count/len(successful_tests)*100:.1f}%)")
        print(f"  SLAV Minimization: {slav_minimized_count}/{len(successful_tests)} tests ({slav_minimized_count/len(successful_tests)*100:.1f}%)")
    
    if failed_tests:
        print(f"\nFailed Tests:")
        print("-" * 80)
        for result in failed_tests:
            print(f"  Test {result['test_case']}: {result['name']}")
            if 'error' in result:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Could not extract weight from response")
    
    # Save results
    output_file = "./planetlab_test_results.json"
    print(f"\n4. Saving results to {output_file}...")
    try:
        # Convert numpy types to Python native types for JSON
        json_results = []
        for r in results:
            json_result = {}
            for key, value in r.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    json_result[key] = float(value)
                elif isinstance(value, np.generic):
                    json_result[key] = value.item()
                elif isinstance(value, dict):
                    # Recursively convert dict values
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
        print(f"   ✓ Results saved successfully to {output_file}")
        
        # In Colab, offer to download results
        if IN_COLAB:
            print(f"\n💡 To download results in Colab, run:")
            print(f"   from google.colab import files")
            print(f"   files.download('{output_file}')")
            
    except Exception as e:
        print(f"   ⚠ Warning: Could not save results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Total tests: {len(results)}")
    print(f"  - Successful: {len(successful_tests)}")
    print(f"  - Failed: {len(failed_tests)}")
    if successful_tests:
        print(f"  - Average weight: {np.mean([r['predicted_w'] for r in successful_tests]):.4f}")
        print(f"  - Profit maximization: {sum(1 for r in successful_tests if r.get('profit_maximized', False))}/{len(successful_tests)} tests")
        print(f"  - SLAV minimization: {sum(1 for r in successful_tests if r.get('slav_minimized', False))}/{len(successful_tests)} tests")
    print("=" * 80)


if __name__ == "__main__":
    main()

