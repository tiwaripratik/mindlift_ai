# -*- coding: utf-8 -*-
"""
Test script for the empathetic response generation model (finetuning2.py)

This script loads the fine-tuned model and tests it with various contexts
to evaluate empathetic response generation.
"""

import os
import sys
import json
import torch
from typing import Dict, List, Optional

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Install required packages if in Colab
if IN_COLAB:
    print("Checking and installing required packages...")
    import subprocess
    
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
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                f"{package_name}{version}", "-q", "--upgrade"
            ])

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    logging,
)
from peft import PeftModel

# Suppress warnings
logging.set_verbosity(logging.ERROR)

# PHQ-8 detection (simplified version for testing)
PHQ8_QUESTIONS = {
    "interest_pleasure": {
        "keywords": ["interest", "pleasure", "enjoy", "fun", "hobby"],
        "negative_indicators": ["don't enjoy", "lost interest", "nothing interests"]
    },
    "mood": {
        "keywords": ["feeling", "down", "depressed", "hopeless", "sad", "mood"],
        "negative_indicators": ["feeling down", "depressed", "hopeless", "worthless"]
    },
    "sleep": {
        "keywords": ["sleep", "insomnia", "rest", "tired", "bed", "wake up"],
        "negative_indicators": ["can't sleep", "wake up", "insomnia", "trouble sleeping"]
    },
    "energy": {
        "keywords": ["energy", "tired", "exhausted", "fatigue", "drained"],
        "negative_indicators": ["no energy", "exhausted", "fatigue", "drained"]
    },
    "self_worth": {
        "keywords": ["myself", "failure", "worthless", "disappointed", "let down"],
        "negative_indicators": ["hate myself", "failure", "worthless", "let everyone down"]
    },
    "concentration": {
        "keywords": ["focus", "concentrate", "attention", "distracted", "think"],
        "negative_indicators": ["can't focus", "can't concentrate", "mind wanders"]
    },
    "psychomotor": {
        "keywords": ["slow", "restless", "fidgety", "moving", "agitated"],
        "negative_indicators": ["moving slow", "can't sit still", "restless"]
    }
}


def assess_depression_level(context: str) -> Dict:
    """Assess depression level from context"""
    detected_topics = []
    total_score = 0
    
    context_lower = context.lower()
    
    for topic, info in PHQ8_QUESTIONS.items():
        negative_indicators = info.get("negative_indicators", [])
        negative_count = sum(1 for indicator in negative_indicators if indicator in context_lower)
        
        if negative_count > 0:
            base_score = min(negative_count, 2)
            intensity_words = ["very", "really", "extremely", "completely", "all the time"]
            if any(word in context_lower for word in intensity_words):
                base_score = min(base_score + 1, 3)
            
            detected_topics.append({
                "topic": topic,
                "score": base_score,
                "confidence": 0.7
            })
            total_score += base_score
    
    # Determine severity
    if total_score <= 4:
        severity = "Minimal"
    elif total_score <= 9:
        severity = "Mild"
    elif total_score <= 14:
        severity = "Moderate"
    elif total_score <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"
    
    return {
        "severity": severity,
        "phq8_score": total_score,
        "detected_symptoms": detected_topics,
        "confidence": 0.7 if detected_topics else 0.5
    }


def load_model(model_path: str = "./final_model_empathetic"):
    """Load the fine-tuned empathetic response model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Please run finetuning2.py first to train the model."
        )
    
    print(f"Loading model from {model_path}...")
    
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge LoRA weights for inference
    
    model.eval()
    
    print("✓ Model loaded successfully")
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    context: str,
    depression_level: Optional[Dict] = None,
    max_length: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """
    Generate empathetic response for given context
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        context: User's statement/context
        depression_level: Optional depression assessment dict (if None, will assess automatically)
        max_length: Maximum response length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    """
    # Assess depression level if not provided
    if depression_level is None:
        depression_level = assess_depression_level(context)
    
    # Build symptoms description
    symptoms_desc = ""
    if depression_level["detected_symptoms"]:
        symptoms_list = [
            f"- {t['topic'].replace('_', ' ').title()}: Score {t['score']}"
            for t in depression_level["detected_symptoms"]
        ]
        symptoms_desc = "\n".join(symptoms_list)
    else:
        symptoms_desc = "- No specific PHQ-8 symptoms detected (general mental health concerns)"
    
    # Create prompt (same format as training)
    prompt = f"""You are an empathetic mental health counselor. A person is seeking help with the following concerns:

Person's Statement:
{context}

Depression Assessment (PHQ-8):
- Severity Level: {depression_level['severity']} Depression
- PHQ-8 Score: {depression_level['phq8_score']}/24
- Detected Symptoms:
{symptoms_desc}

Please provide a supportive, empathetic, and helpful response. Acknowledge their feelings, validate their experience, and offer constructive guidance. Be warm, understanding, and professional."""
    
    # Tokenize
    inputs = tokenizer(
        f"[INST] {prompt} [/INST]",
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part (after [/INST])
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()
    
    return response


def run_test_cases(model, tokenizer):
    """Run predefined test cases"""
    
    test_cases = [
        {
            "name": "Severe Depression - Worthlessness",
            "context": "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here. I've never tried or contemplated suicide. I've always wanted to fix my issues, but I never get around to it.",
            "expected_traits": ["empathetic", "supportive", "validating", "professional"]
        },
        {
            "name": "Moderate Depression - Sleep Issues",
            "context": "I've been having trouble sleeping for the past few weeks. I can't fall asleep until 3 AM and then I wake up exhausted. I feel tired all the time and have no energy to do anything.",
            "expected_traits": ["understanding", "practical", "helpful"]
        },
        {
            "name": "Mild Depression - Loss of Interest",
            "context": "I used to love reading and going for walks, but lately I don't enjoy anything. Nothing seems fun anymore. I still do things but I don't get pleasure from them.",
            "expected_traits": ["empathetic", "encouraging"]
        },
        {
            "name": "Minimal Depression - General Concern",
            "context": "I've been feeling a bit down lately. Nothing serious, but I've been noticing I'm not as happy as I used to be. I'm wondering if I should talk to someone about it.",
            "expected_traits": ["supportive", "validating"]
        },
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING TEST CASES")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'='*80}")
        
        print(f"\n📝 Context:")
        print(f"{test_case['context']}")
        
        # Assess depression level
        assessment = assess_depression_level(test_case['context'])
        print(f"\n🔍 Depression Assessment:")
        print(f"   Severity: {assessment['severity']} Depression")
        print(f"   PHQ-8 Score: {assessment['phq8_score']}/24")
        if assessment['detected_symptoms']:
            print(f"   Detected Symptoms:")
            for symptom in assessment['detected_symptoms']:
                print(f"     - {symptom['topic'].replace('_', ' ').title()}: Score {symptom['score']}")
        
        # Generate response
        print(f"\n🤖 Generated Response:")
        print("-" * 80)
        try:
            response = generate_response(
                model, tokenizer, test_case['context'],
                depression_level=assessment,
                max_length=500,
                temperature=0.7
            )
            print(response)
            print("-" * 80)
            
            # Evaluate response quality
            response_lower = response.lower()
            traits_found = []
            for trait in test_case['expected_traits']:
                if trait in response_lower or any(
                    word in response_lower 
                    for word in ['support', 'empath', 'understand', 'help', 'care']
                ):
                    traits_found.append(trait)
            
            results.append({
                "test_case": test_case['name'],
                "success": len(traits_found) > 0,
                "response_length": len(response),
                "response": response
            })
            
            print(f"\n✓ Response generated ({len(response)} characters)")
            
        except Exception as e:
            print(f"✗ Error generating response: {e}")
            results.append({
                "test_case": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r.get('success', False))
    print(f"Successful: {successful}/{len(test_cases)}")
    print(f"Failed: {len(test_cases) - successful}/{len(test_cases)}")
    
    return results


def interactive_mode(model, tokenizer):
    """Interactive mode for testing the model"""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter user contexts/questions. Type 'quit' or 'exit' to stop.")
    print()
    
    while True:
        try:
            context = input("User Context: ").strip()
            
            if context.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode...")
                break
            
            if not context:
                continue
            
            print("\n🔍 Assessing depression level...")
            assessment = assess_depression_level(context)
            
            print(f"\nDepression Assessment:")
            print(f"   Severity: {assessment['severity']} Depression")
            print(f"   PHQ-8 Score: {assessment['phq8_score']}/24")
            
            print("\n🤖 Generating empathetic response...")
            response = generate_response(
                model, tokenizer, context,
                depression_level=assessment,
                max_length=500,
                temperature=0.7
            )
            
            print("\n" + "-" * 80)
            print("RESPONSE:")
            print("-" * 80)
            print(response)
            print("-" * 80)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


def main():
    """Main function"""
    print("=" * 80)
    print("Empathetic Response Model Tester")
    print("=" * 80)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("⚠ Using CPU (slower)")
    
    # Run test cases
    print("\nSelect mode:")
    print("1. Run predefined test cases")
    print("2. Interactive mode")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3, default=1): ").strip() or "1"
    
    if choice in ["1", "3"]:
        results = run_test_cases(model, tokenizer)
        
        # Save results
        with open("test_results_empathetic.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Test results saved to test_results_empathetic.json")
    
    if choice in ["2", "3"]:
        interactive_mode(model, tokenizer)
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

