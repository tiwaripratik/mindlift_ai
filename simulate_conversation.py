# -*- coding: utf-8 -*-
"""
Simulated Conversation Test for Fine-tuned Model
COLAB-COMPATIBLE

This script simulates a full conversation where predefined user responses
test the model's depression detection capabilities.

Usage:
    # In Colab:
    !python simulate_conversation.py
    
    # Or locally:
    python simulate_conversation.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable bitsandbytes by default
USE_4BIT = False

# Model paths
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./final_model"

# PHQ-8 Questions
PHQ8_QUESTIONS = {
    "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
    "mood": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
    "sleep": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
    "energy": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
    "appetite": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
    "self_worth": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
    "concentration": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
    "psychomotor": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?"
}

# Simulated conversations with different depression levels
SIMULATED_CONVERSATIONS = [
    {
        "name": "Case 1: Moderate Depression",
        "conversation": [
            {
                "speaker": "Interviewer",
                "text": "Hi! How have you been feeling lately?"
            },
            {
                "speaker": "User",
                "text": "I've been feeling really down and hopeless. Nothing seems to matter anymore."
            },
            {
                "speaker": "Interviewer",
                "text": "Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?"
            },
            {
                "speaker": "User",
                "text": "Yes, I've completely lost interest in everything. I used to love reading and going for walks, but now I don't want to do anything."
            },
            {
                "speaker": "Interviewer",
                "text": "How have you been sleeping?"
            },
            {
                "speaker": "User",
                "text": "Terrible. I have trouble falling asleep almost every night, and when I do sleep, I wake up early and can't get back to sleep."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling tired or having little energy?"
            },
            {
                "speaker": "User",
                "text": "I'm exhausted all the time. Even small tasks feel impossible. I have no energy for anything."
            },
            {
                "speaker": "Interviewer",
                "text": "How's your appetite been?"
            },
            {
                "speaker": "User",
                "text": "I've lost my appetite completely. I have to force myself to eat, and I've lost weight."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling bad about yourself recently?"
            },
            {
                "speaker": "User",
                "text": "Yes, I feel like a failure. I think I've let everyone down - my family, my friends, myself."
            },
        ]
    },
    {
        "name": "Case 2: Low Depression (Healthy)",
        "conversation": [
            {
                "speaker": "Interviewer",
                "text": "Hi! How have you been feeling lately?"
            },
            {
                "speaker": "User",
                "text": "I've been feeling pretty good actually. Life has been going well."
            },
            {
                "speaker": "Interviewer",
                "text": "Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?"
            },
            {
                "speaker": "User",
                "text": "No, not really. I still enjoy my hobbies and spending time with friends. I'm looking forward to the weekend."
            },
            {
                "speaker": "Interviewer",
                "text": "How have you been sleeping?"
            },
            {
                "speaker": "User",
                "text": "Sleep has been fine. I usually get 7-8 hours and feel rested in the morning."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling tired or having little energy?"
            },
            {
                "speaker": "User",
                "text": "I have normal energy levels. I work out regularly and feel good about my routine."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling down, depressed, or hopeless?"
            },
            {
                "speaker": "User",
                "text": "No, I haven't been feeling depressed. I have my ups and downs like everyone, but overall I'm doing well."
            },
        ]
    },
    {
        "name": "Case 3: Severe Depression",
        "conversation": [
            {
                "speaker": "Interviewer",
                "text": "Hi! How have you been feeling lately?"
            },
            {
                "speaker": "User",
                "text": "I don't know why I'm even here. Everything feels pointless. I can't see any hope."
            },
            {
                "speaker": "Interviewer",
                "text": "Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?"
            },
            {
                "speaker": "User",
                "text": "I've lost all interest in everything. Nothing brings me pleasure anymore. I don't even want to get out of bed."
            },
            {
                "speaker": "Interviewer",
                "text": "How have you been sleeping?"
            },
            {
                "speaker": "User",
                "text": "I can't sleep at all. I lie awake all night thinking about how worthless I am. When I do sleep, I have nightmares."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling tired or having little energy?"
            },
            {
                "speaker": "User",
                "text": "I'm completely drained. I can barely move. Everything takes so much effort. I feel like I'm moving through mud."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling bad about yourself recently?"
            },
            {
                "speaker": "User",
                "text": "I hate myself. I'm a complete failure. I've ruined everything and everyone would be better off without me."
            },
            {
                "speaker": "Interviewer",
                "text": "How's your ability to concentrate been?"
            },
            {
                "speaker": "User",
                "text": "I can't focus on anything. My mind is blank or racing. I can't even read a simple paragraph without losing track."
            },
        ]
    },
    {
        "name": "Case 4: Mild Depression (Mixed)",
        "conversation": [
            {
                "speaker": "Interviewer",
                "text": "Hi! How have you been feeling lately?"
            },
            {
                "speaker": "User",
                "text": "I've been a bit down lately. Nothing serious, but I've been feeling a bit off."
            },
            {
                "speaker": "Interviewer",
                "text": "Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?"
            },
            {
                "speaker": "User",
                "text": "Sometimes I don't feel like doing things I used to enjoy, but I still do them occasionally and find some pleasure."
            },
            {
                "speaker": "Interviewer",
                "text": "How have you been sleeping?"
            },
            {
                "speaker": "User",
                "text": "Sleep has been okay, but I've had a few nights where I had trouble falling asleep. Maybe 2-3 times in the past couple weeks."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling tired or having little energy?"
            },
            {
                "speaker": "User",
                "text": "I've been feeling a bit more tired than usual. Not exhausted, but I do feel like I need more rest."
            },
            {
                "speaker": "Interviewer",
                "text": "Have you been feeling down, depressed, or hopeless?"
            },
            {
                "speaker": "User",
                "text": "I've had some days where I feel down, but it's not constant. I still have hope and good days mixed in."
            },
        ]
    },
]


def load_model():
    """Load the fine-tuned model"""
    print("=" * 80)
    print("Loading Fine-tuned Model")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}\n"
            "Please ensure the model has been trained and saved to ./final_model"
        )
    
    print(f"\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"2. Loading base model...")
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device_map}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    
    print("3. Loading LoRA adapter...")
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model = model.merge_and_unload()
    
    print("✓ Model loaded successfully!\n")
    return model, tokenizer


def format_assessment_prompt(conversation_text: str) -> str:
    """Format prompt for depression assessment - MUST MATCH TRAINING FORMAT"""
    # Match the exact format from finetuning.py line 847
    return f"""Analyze the following conversation for depression indicators using PHQ-8 assessment:

Conversation:
{conversation_text}

Identify:
1. PHQ-8 symptoms present in the user's responses
2. PHQ-8 scores for each detected symptom
3. Overall depression assessment"""


def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate response from the model"""
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
    else:
        response = generated_text[len(formatted_prompt):].strip()
    
    return response


def simulate_conversation(model, tokenizer, conversation_data: Dict):
    """Simulate a conversation and get assessment"""
    print("\n" + "=" * 80)
    print(f"CONVERSATION: {conversation_data['name']}")
    print("=" * 80)
    
    # Display conversation
    print("\n📝 Conversation:")
    print("-" * 80)
    conversation_text = ""
    for turn in conversation_data['conversation']:
        print(f"{turn['speaker']}: {turn['text']}")
        conversation_text += f"{turn['speaker']}: {turn['text']}\n"
    print("-" * 80)
    
    # Get assessment
    print("\n🤖 Model: Analyzing conversation for depression indicators...\n")
    prompt = format_assessment_prompt(conversation_text.strip())
    
    try:
        assessment = generate_response(model, tokenizer, prompt, max_length=600)
        
        print("=" * 80)
        print("DEPRESSION ASSESSMENT:")
        print("=" * 80)
        print(assessment)
        print("=" * 80)
        
        return assessment
        
    except Exception as e:
        print(f"❌ Error generating assessment: {e}")
        return None


def main():
    """Main function"""
    print("=" * 80)
    print("Simulated Conversation Test - Depression Detection")
    print("=" * 80)
    print("\nThis script simulates multiple conversations with different")
    print("depression levels to test the model's detection capabilities.\n")
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    # Run simulations
    results = []
    for i, conv_data in enumerate(SIMULATED_CONVERSATIONS, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/{len(SIMULATED_CONVERSATIONS)}")
        print(f"{'='*80}")
        
        assessment = simulate_conversation(model, tokenizer, conv_data)
        
        results.append({
            "case": conv_data['name'],
            "conversation": conv_data['conversation'],
            "assessment": assessment,
            "status": "success" if assessment else "error"
        })
        
        # Small delay between tests
        import time
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nTotal test cases: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")
    
    # Save results
    output_file = "simulated_conversation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

