# -*- coding: utf-8 -*-
"""
Quick Test Script for Fine-tuned Model (Colab-Friendly)
Simpler version for quick testing in Colab

Usage in Colab:
    # Run this cell after training
    !python quick_test.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./final_model"

# Test prompt
test_prompt = """Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
Interviewer: Over the last 2 weeks, have you been bothered by little interest or pleasure in doing things?
User: I've completely lost interest in everything. Nothing seems fun anymore.

PHQ-8 Questions Reference:
{
  "interest_pleasure": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?"
}"""

print("=" * 80)
print("Quick Model Test")
print("=" * 80)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ Model not found at: {MODEL_PATH}")
    print("Please train the model first using finetuning.py")
    exit(1)

print(f"\n1. Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

print(f"2. Loading base model: {BASE_MODEL}...")
device_map = "auto" if torch.cuda.is_available() else "cpu"
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)

print(f"3. Loading LoRA adapter from {MODEL_PATH}...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model = model.merge_and_unload()

print("4. Model loaded! Generating response...\n")

# Format prompt
formatted_prompt = f"### Instruction:\n{test_prompt}\n\n### Response:\n"

# Tokenize and generate
inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_text = response.split("### Response:\n")[-1].strip()

print("=" * 80)
print("RESPONSE:")
print("=" * 80)
print(response_text)
print("=" * 80)

