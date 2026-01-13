# -*- coding: utf-8 -*-
"""
Fine-tuning script for Mistral 7B to generate empathetic responses
based on conversation context and depression level assessment.

This script:
1. Processes JSON data with Context/Response pairs
2. Assesses depression level from Context using PHQ-8 detection
3. Creates input/output pairs for empathetic response training
4. Fine-tunes Mistral 7B using LoRA

COLAB SETUP INSTRUCTIONS:
=========================
OPTION 1 (Recommended): Install packages first, then run script
---------------------------------------------------------------
# Run this in a Colab cell first:
!pip install -q transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch

# Then upload your JSON file:
from google.colab import files
uploaded = files.upload()

# Then run this script:
!python finetuning2.py

OPTION 2: Let script auto-install (may take longer)
---------------------------------------------------
# Just copy/paste this entire file into a Colab cell and run it.
"""

import os
import sys
import subprocess
import json
import re
from typing import List, Dict, Tuple, Optional

# Set PyTorch CUDA memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Quantization setting - MUST be defined before imports to avoid bitsandbytes issues
USE_4BIT = False  # Set to False to disable quantization and avoid bitsandbytes issues (RECOMMENDED)

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
        ('torch', '>=2.0.0')
    ]
    
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
                print(f"✓ Installed {package_name}")
            except Exception as e:
                print(f"⚠ Warning: Could not install {package_name}: {e}")

# Now import the packages
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
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    else:
        from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
except ImportError as e:
    print("=" * 80)
    print("ERROR: Missing required packages!")
    print("=" * 80)
    print(f"Failed to import: {e}")
    sys.exit(1)

# Import PHQ-8 detection functions from finetuning.py
# We'll reuse the depression detection logic
PHQ8_QUESTIONS = {
    "interest_pleasure": {
        "question": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things?",
        "keywords": ["interest", "pleasure", "enjoy", "fun", "hobby", "excited", "looking forward", "activities"],
        "negative_indicators": ["don't enjoy", "lost interest", "nothing interests", "pointless", "stopped doing", "no fun"]
    },
    "mood": {
        "question": "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
        "keywords": ["feeling", "down", "depressed", "hopeless", "sad", "mood", "empty", "numb"],
        "negative_indicators": ["feeling down", "depressed", "hopeless", "worthless", "empty", "sad"]
    },
    "sleep": {
        "question": "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
        "keywords": ["sleep", "insomnia", "rest", "tired", "bed", "wake up", "sleeping", "awake"],
        "negative_indicators": ["can't sleep", "wake up", "insomnia", "too much sleep", "oversleeping", "trouble sleeping"]
    },
    "energy": {
        "question": "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
        "keywords": ["energy", "tired", "exhausted", "fatigue", "drained", "worn out", "lethargic"],
        "negative_indicators": ["no energy", "exhausted", "fatigue", "drained", "can barely", "tired all the time"]
    },
    "appetite": {
        "question": "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
        "keywords": ["eat", "appetite", "food", "hungry", "meal", "weight", "eating"],
        "negative_indicators": ["no appetite", "not hungry", "forcing myself", "eating too much", "can't stop eating"]
    },
    "self_worth": {
        "question": "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
        "keywords": ["myself", "failure", "worthless", "disappointed", "let down", "useless", "self"],
        "negative_indicators": ["hate myself", "failure", "worthless", "let everyone down", "disappointed", "bad about myself"]
    },
    "concentration": {
        "question": "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things, such as reading the newspaper or watching television?",
        "keywords": ["focus", "concentrate", "attention", "distracted", "think", "remember", "reading"],
        "negative_indicators": ["can't focus", "can't concentrate", "mind wanders", "forget everything", "distracted"]
    },
    "psychomotor": {
        "question": "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
        "keywords": ["slow", "restless", "fidgety", "moving", "agitated", "sluggish", "pace"],
        "negative_indicators": ["moving slow", "can't sit still", "restless", "agitated", "everything takes forever"]
    }
}

PHQ8_SCORE_MAPPING = {
    0: {
        "patterns": ["not at all", "never", "no", "haven't", "not really", "rarely", "none"],
    },
    1: {
        "patterns": ["sometimes", "occasionally", "few times", "couple days", "once in a while", "several days"],
    },
    2: {
        "patterns": ["often", "most days", "frequently", "more than half", "usually", "majority of time"],
    },
    3: {
        "patterns": ["always", "every day", "constantly", "all the time", "daily", "nearly every day"],
    }
}


def extract_phq8_answer(response_text: str, topic: str) -> Tuple[Optional[int], float]:
    """Extract PHQ-8 answer score (0-3) from user response"""
    response_lower = response_text.lower().strip()
    confidence = 0.0
    
    topic_info = PHQ8_QUESTIONS.get(topic, {})
    negative_indicators = topic_info.get("negative_indicators", [])
    keywords = topic_info.get("keywords", [])
    
    # Check for negative indicators
    negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
    
    # Check for explicit frequency indicators
    for score, score_data in PHQ8_SCORE_MAPPING.items():
        patterns = score_data.get("patterns", [])
        for pattern in patterns:
            if pattern in response_lower:
                if negative_count > 0:
                    return score, 0.85
                return score, 0.7
    
    # Use negative indicators to estimate severity
    if negative_count > 0:
        base_score = min(negative_count, 2)
        intensity_words = ["very", "really", "extremely", "completely", "totally", "all the time"]
        has_high_intensity = any(word in response_lower for word in intensity_words)
        
        if has_high_intensity:
            base_score = min(base_score + 1, 3)
        
        confidence = 0.6 + (negative_count * 0.1)
        return base_score, min(confidence, 0.95)
    
    return None, 0.0


def assess_depression_level(context: str) -> Dict:
    """
    Assess depression level from context using PHQ-8 detection
    
    Returns:
        {
            "severity": str,  # "Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"
            "phq8_score": int,  # 0-24
            "detected_symptoms": List[Dict],  # List of {topic, score, confidence}
            "confidence": float  # Overall confidence
        }
    """
    detected_topics = []
    total_score = 0
    
    context_lower = context.lower()
    
    # Check for depression indicators across all PHQ-8 topics
    for topic, info in PHQ8_QUESTIONS.items():
        answer_score, confidence = extract_phq8_answer(context, topic)
        if answer_score is not None and confidence > 0.3:
            detected_topics.append({
                "topic": topic,
                "score": answer_score,
                "confidence": confidence
            })
            total_score += answer_score
    
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
    
    # Calculate overall confidence
    overall_confidence = sum(t["confidence"] for t in detected_topics) / len(detected_topics) if detected_topics else 0.5
    
    return {
        "severity": severity,
        "phq8_score": total_score,
        "detected_symptoms": detected_topics,
        "confidence": overall_confidence
    }


def load_json_data(json_path: str) -> List[Dict]:
    """Load JSON data with Context/Response pairs - handles JSON, JSONL, and NDJSON formats"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    data = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # Read first few lines to detect format
            first_lines = []
            for i, line in enumerate(f):
                if i < 3:  # Read first 3 lines
                    first_lines.append(line.strip())
                else:
                    break
            
            # Reset file pointer
            f.seek(0)
            
            # Check if it looks like JSONL (multiple JSON objects, one per line)
            is_jsonl = False
            if len(first_lines) > 1:
                # Check if first line is a valid JSON object
                try:
                    json.loads(first_lines[0])
                    # If second line also looks like JSON, it's likely JSONL
                    if len(first_lines) > 1 and first_lines[1]:
                        try:
                            json.loads(first_lines[1])
                            is_jsonl = True
                        except:
                            pass
                except:
                    pass
            
            # If JSONL format detected, parse line by line
            if is_jsonl:
                print("   Detected JSONL format (one JSON object per line)")
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            data.append(parsed)
                        else:
                            print(f"   Warning: Line {line_num} is not a JSON object, skipping")
                    except json.JSONDecodeError as e:
                        print(f"   Warning: Failed to parse line {line_num}: {e}")
                        continue
            else:
                # Try parsing as regular JSON (single object or array)
                content = f.read().strip()
                if not content:
                    raise ValueError("JSON file is empty")
                
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        data = [parsed]
                    elif isinstance(parsed, list):
                        data = parsed
                    else:
                        raise ValueError(f"JSON must be an object or array, got {type(parsed)}")
                except json.JSONDecodeError as e:
                    # If that fails, try JSONL format as fallback
                    print(f"   Standard JSON parsing failed: {e}")
                    print("   Attempting JSONL format (one JSON object per line)...")
                    f.seek(0)
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict):
                                data.append(parsed)
                            else:
                                print(f"   Warning: Line {line_num} is not a JSON object, skipping")
                        except json.JSONDecodeError:
                            print(f"   Warning: Failed to parse line {line_num}, skipping")
                            continue
    except Exception as e:
        raise ValueError(f"Failed to load JSON file: {e}")
    
    if not data:
        raise ValueError("No valid JSON data found in file")
    
    print(f"   Successfully loaded {len(data)} entries")
    return data


def process_to_training_pairs(data: List[Dict]) -> List[Dict]:
    """
    Process JSON data into input/output pairs for empathetic response training
    
    Format:
    - Input: Context + Depression Level Assessment
    - Output: Empathetic Response
    """
    training_pairs = []
    
    stats = {
        "total_entries": 0,
        "processed": 0,
        "minimal_depression": 0,
        "mild_depression": 0,
        "moderate_depression": 0,
        "moderately_severe_depression": 0,
        "severe_depression": 0,
    }
    
    for entry in data:
        stats["total_entries"] += 1
        
        # Extract Context and Response
        context = entry.get("Context", entry.get("context", ""))
        response = entry.get("Response", entry.get("response", ""))
        
        if not context or not response:
            continue
        
        # Assess depression level from context
        assessment = assess_depression_level(context)
        
        # Update statistics
        severity = assessment["severity"]
        if severity == "Minimal":
            stats["minimal_depression"] += 1
        elif severity == "Mild":
            stats["mild_depression"] += 1
        elif severity == "Moderate":
            stats["moderate_depression"] += 1
        elif severity == "Moderately Severe":
            stats["moderately_severe_depression"] += 1
        elif severity == "Severe":
            stats["severe_depression"] += 1
        
        # Build detected symptoms description
        symptoms_desc = ""
        if assessment["detected_symptoms"]:
            symptoms_list = [
                f"- {t['topic'].replace('_', ' ').title()}: Score {t['score']}"
                for t in assessment["detected_symptoms"]
            ]
            symptoms_desc = "\n".join(symptoms_list)
        else:
            symptoms_desc = "- No specific PHQ-8 symptoms detected (general mental health concerns)"
        
        # Create input text with context and depression assessment
        input_text = f"""You are an empathetic mental health counselor. A person is seeking help with the following concerns:

Person's Statement:
{context}

Depression Assessment (PHQ-8):
- Severity Level: {severity} Depression
- PHQ-8 Score: {assessment['phq8_score']}/24
- Detected Symptoms:
{symptoms_desc}

Please provide a supportive, empathetic, and helpful response. Acknowledge their feelings, validate their experience, and offer constructive guidance. Be warm, understanding, and professional."""
        
        # The output is the empathetic response
        output_text = response.strip()
        
        training_pairs.append({
            "input": input_text,
            "output": output_text,
            "severity": severity,
            "phq8_score": assessment["phq8_score"],
            "confidence": assessment["confidence"]
        })
        
        stats["processed"] += 1
    
    # Print statistics
    print(f"\n   Processing Statistics:")
    print(f"   - Total entries: {stats['total_entries']}")
    print(f"   - Processed: {stats['processed']}")
    print(f"   - Minimal depression: {stats['minimal_depression']}")
    print(f"   - Mild depression: {stats['mild_depression']}")
    print(f"   - Moderate depression: {stats['moderate_depression']}")
    print(f"   - Moderately severe depression: {stats['moderately_severe_depression']}")
    print(f"   - Severe depression: {stats['severe_depression']}")
    print(f"   - Total training pairs created: {len(training_pairs)}")
    
    return training_pairs


def save_training_data(training_pairs: List[Dict], output_path: str):
    """Save training data to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(training_pairs)} training pairs to {output_path}")


# LoRA parameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Training parameters
# NOTE: Optimized to use ~65GB GPU memory (stay within limit)
OUTPUT_DIR = "./results_empathetic"
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 18  # Optimized for ~65GB GPU usage
PER_DEVICE_EVAL_BATCH_SIZE = 18
GRADIENT_ACCUMULATION_STEPS = 3  # Effective batch size: 54
GRADIENT_CHECKPOINTING = True  # Enabled to reduce memory usage
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = False
SAVE_STEPS = 500
LOGGING_STEPS = 25
MAX_SEQ_LENGTH = 2048  # Optimized for ~65GB GPU usage
PACKING = False


def formatting_func(example):
    """Format training examples for Mistral"""
    return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"


def main():
    """Main training function - COLAB COMPATIBLE"""
    print("=" * 80)
    print("Mistral 7B Fine-tuning for Empathetic Response Generation")
    print("=" * 80)
    
    # Step 1: Load JSON data
    json_path = None
    possible_paths = [
        "training_data.json",
        "data.json",
        "/content/training_data.json",
        "/content/data.json",
        "/content/drive/MyDrive/training_data.json",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            json_path = path
            break
    
    if json_path is None:
        print("\nJSON file not found in common locations.")
        print("Please upload your JSON file or specify the path.")
        print("\nIn Colab, you can:")
        print("1. Upload file: from google.colab import files; files.upload()")
        print("2. Or mount Drive: from google.colab import drive; drive.mount('/content/drive')")
        print("\nTrying to list JSON files in /content...")
        try:
            if os.path.exists("/content"):
                json_files = [f for f in os.listdir("/content") if f.endswith(".json")]
                if json_files:
                    print(f"Found JSON files: {json_files}")
                    json_path = f"/content/{json_files[0]}"
                    print(f"Using: {json_path}")
                else:
                    print("No JSON files found. Please upload your data file.")
                    return
        except Exception as e:
            print(f"Error checking /content: {e}")
            return
    
    print(f"\n1. Loading JSON data from {json_path}...")
    data = load_json_data(json_path)
    print(f"   Loaded {len(data)} entries")
    
    # Step 2: Process into training pairs
    print("\n2. Processing data into training pairs...")
    training_pairs = process_to_training_pairs(data)
    print(f"   Created {len(training_pairs)} training pairs")
    
    if len(training_pairs) == 0:
        print("ERROR: No training pairs created. Please check your data format.")
        return
    
    # Step 3: Save training data
    training_data_path = "empathetic_training_data.json"
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
    
    # Load model without quantization (USE_4BIT = False)
    print("   Loading model without quantization...")
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
    
    print("   ✓ Successfully loaded model")
    
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
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=PACKING,
    )
    
    # Step 9: Train model
    print("\n9. Starting training...")
    print("=" * 80)
    
    # Print GPU utilization info
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
        print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} (effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
        print(f"   Max Sequence Length: {MAX_SEQ_LENGTH}")
        print(f"   Gradient Checkpointing: {GRADIENT_CHECKPOINTING}")
    
    print("\n📊 Training progress will be displayed below:")
    print("=" * 80 + "\n")
    
    # Aggressive memory clearing before training
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
        if "out of memory" in error_str or "cuda" in error_str:
            print("\n" + "=" * 80)
            print("ERROR: GPU Out of Memory")
            print("=" * 80)
            print("\nTry reducing:")
            print(f"  - PER_DEVICE_TRAIN_BATCH_SIZE (currently {PER_DEVICE_TRAIN_BATCH_SIZE})")
            print(f"  - MAX_SEQ_LENGTH (currently {MAX_SEQ_LENGTH})")
            print("  - Enable gradient checkpointing")
            raise
        else:
            raise
    
    # Step 10: Save model
    print("\n10. Saving fine-tuned model...")
    trainer.model.save_pretrained("./final_model_empathetic")
    tokenizer.save_pretrained("./final_model_empathetic")
    print("   Model saved to ./final_model_empathetic")
    
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

if os.path.exists('./final_model_empathetic'):
    shutil.make_archive('final_model_empathetic', 'zip', './final_model_empathetic')
    files.download('final_model_empathetic.zip')
    print("Model download started!")
        """)
    
    # Test the model
    print("\nTesting the model...")
    logging.set_verbosity(logging.CRITICAL)
    
    test_context = "I'm going through some things with my feelings and myself. I barely sleep and I do nothing but think about how I'm worthless and how I shouldn't be here."
    
    prompt = f"""You are an empathetic mental health counselor. A person is seeking help with the following concerns:

Person's Statement:
{test_context}

Depression Assessment (PHQ-8):
- Severity Level: Moderate Depression
- PHQ-8 Score: 12/24
- Detected Symptoms:
- Sleep: Score 3
- Self Worth: Score 3
- Mood: Score 2

Please provide a supportive, empathetic, and helpful response."""
    
    try:
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=500,
            temperature=0.7,
        )
        result = pipe(f"[INST] {prompt} [/INST]")
        print("\nTest Output:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"\nNote: Could not run test inference: {e}")
        print("Model training completed successfully. You can load and test the model separately.")


if __name__ == "__main__":
    main()

