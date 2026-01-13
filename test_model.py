# -*- coding: utf-8 -*-
"""
Test script for fine-tuned Mistral 7B PHQ-8 Depression Detection Model
COLAB-COMPATIBLE

This script loads the fine-tuned model from ./final_model and tests it with
various PHQ-8 related inputs.

Usage:
    # In Colab:
    !python test_model.py
    
    # Or locally:
    python test_model.py

Requirements:
    - The model must be trained and saved to ./final_model directory
    - Required packages: transformers, peft, torch

What it tests:
    1. PHQ-8 question identification
    2. PHQ-8 score extraction (0-3)
    3. Depression risk assessment
    4. General depression detection from conversations

Output:
    - Prints test results to console
    - Saves detailed results to ./test_results.json
"""

import os
import sys
import json
import torch
from typing import List, Dict

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
    ]
    
    for package_name, version in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            print(f"Installing {package_name}{version}...")
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}{version}", "-q", "--upgrade"
            ])
            print(f"✓ Installed {package_name}")

# Import after checking/installing dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Disable bitsandbytes by default
USE_4BIT = False

# Model paths
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./final_model"

# Test cases
TEST_CASES = [
    {
        "name": "PHQ-8 Interest/Pleasure Question",
        "input": """Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
Interviewer: Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?
User: I've completely lost interest in everything. Nothing seems fun anymore. I don't even want to do things I used to enjoy.

PHQ-8 Questions Reference:
{
  "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
  "mood": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
  "sleep": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
  "energy": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
  "appetite": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
  "self_worth": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
  "concentration": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
  "psychomotor": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
}"""
    },
    {
        "name": "PHQ-8 Sleep Question",
        "input": """Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
Interviewer: How have you been sleeping lately?
User: I've been having trouble falling asleep almost every night. I'm lucky if I get 4 hours of sleep.

PHQ-8 Questions Reference:
{
  "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
  "mood": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
  "sleep": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
  "energy": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
  "appetite": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
  "self_worth": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
  "concentration": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
  "psychomotor": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
}"""
    },
    {
        "name": "General Depression Detection",
        "input": """Analyze the following conversation for depression indicators using PHQ-8 assessment:
Conversation:
Interviewer: How have you been feeling lately?
User: I've been feeling really down and hopeless. I don't have energy for anything, and I can't sleep. I feel like I'm worthless and everything is pointless.

Identify:
1. PHQ-8 symptoms present in the user's responses
2. PHQ-8 scores for each detected symptom
3. Overall depression assessment"""
    },
    {
        "name": "Low Depression Score (Score 0)",
        "input": """Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
Interviewer: Have you been feeling down or depressed lately?
User: No, not really. I've been feeling pretty good actually. I'm enjoying my work and have been spending time with friends.

PHQ-8 Questions Reference:
{
  "mood": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
  "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
  "sleep": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
  "energy": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
  "appetite": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
  "self_worth": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
  "concentration": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
  "psychomotor": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
}"""
    },
]


def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    print("=" * 80)
    print("Loading Fine-tuned Model")
    print("=" * 80)
    
    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}\n"
            "Please ensure the model has been trained and saved to ./final_model"
        )
    
    print(f"\n1. Loading base model: {BASE_MODEL_NAME}")
    print(f"2. Loading fine-tuned adapter from: {MODEL_PATH}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"\n3. Device: {device_map}")
    
    try:
        # Try loading base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        
        print("\n4. Loading LoRA adapter...")
        # Load fine-tuned LoRA adapter
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.merge_and_unload()  # Merge adapter weights for inference
        
        print("   ✓ Model loaded successfully!")
        
        # Check GPU memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024**3)
            print(f"   GPU Memory Available: {free_memory_gb:.2f} GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the model was trained successfully")
        print("2. Check that ./final_model directory exists")
        print("3. Verify all required files are present (adapter_config.json, adapter_model.bin, etc.)")
        raise


def format_prompt(instruction: str) -> str:
    """Format the prompt for Mistral instruction format"""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(model, tokenizer, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
    """Generate response from the model"""
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:\n")
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    return response


def test_model(model, tokenizer):
    """Run test cases on the model"""
    print("\n" + "=" * 80)
    print("Testing Fine-tuned Model")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*80}")
        print(f"\n📝 Input:\n{test_case['input'][:200]}...")
        print("\n🤖 Generating response...")
        
        try:
            response = generate_response(
                model,
                tokenizer,
                test_case['input'],
                max_length=512,
                temperature=0.7
            )
            
            print(f"\n✅ Response:\n{response}")
            
            results.append({
                "test_case": test_case['name'],
                "input": test_case['input'],
                "output": response,
                "status": "success"
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results.append({
                "test_case": test_case['name'],
                "input": test_case['input'],
                "output": None,
                "status": "error",
                "error": str(e)
            })
    
    return results


def save_results(results: List[Dict], output_path: str = "./test_results.json"):
    """Save test results to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Test results saved to: {output_path}")


def main():
    """Main test function"""
    print("=" * 80)
    print("Mistral 7B PHQ-8 Depression Detection - Model Testing")
    print("=" * 80)
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    # Run tests
    try:
        results = test_model(model, tokenizer)
        
        # Summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        print(f"\n✅ Successful: {successful}/{len(results)}")
        if failed > 0:
            print(f"❌ Failed: {failed}/{len(results)}")
        
        # Save results
        save_results(results)
        
        print("\n" + "=" * 80)
        print("Testing Complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        raise


if __name__ == "__main__":
    main()

