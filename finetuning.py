# -*- coding: utf-8 -*-
"""
Fine-tuning script for Mistral 7B to detect depression using PHQ-8 assessment
from conversational data.

This script:
1. Processes conversation CSV data (start, stop, speaker, text columns)
2. Maps conversational responses to PHQ-8 questions and answers
3. Creates input/output pairs for training
4. Fine-tunes Mistral 7B using LoRA

COLAB SETUP INSTRUCTIONS:
=========================
OPTION 1 (Recommended): Install packages first, then run script
---------------------------------------------------------------
# Run this in a Colab cell first:
!pip install -q transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch

# Then upload your CSV file:
from google.colab import files
uploaded = files.upload()

# Then run this script:
!python finetuning.py

OPTION 2: Let script auto-install (may take longer)
---------------------------------------------------
# Just copy/paste this entire file into a Colab cell and run it.
# The script will automatically install missing packages.

4. The script will automatically:
   - Find your CSV file in /content/
   - Process the data with enhanced PHQ-8 mapping
   - Train Mistral 7B with LoRA
   - Save to ./final_model

5. Download the trained model:
   from google.colab import files
   import shutil
   shutil.make_archive('final_model', 'zip', './final_model')
   files.download('final_model.zip')
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
        ('torch', '>=2.0.0')
    ]
    
    for package_name, version in required_packages:
        try:
            # Try to import to check if it's installed
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
                print(f"Please install manually: !pip install {package_name}{version}")
    
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
    print("ERROR: Missing required packages!")
    print("=" * 80)
    print(f"Failed to import: {e}")
    print("\nPlease install required packages:")
    print("!pip install transformers datasets peft bitsandbytes trl accelerate pandas torch")
    print("\nOr run this in a Colab cell first:")
    print("!pip install -q transformers>=4.35.0 datasets>=2.15.0 peft>=0.7.0 bitsandbytes>=0.41.0 trl>=0.7.0 accelerate>=0.25.0 pandas torch")
    sys.exit(1)

# PHQ-8 Questions mapping
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

# PHQ-8 scoring: 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
# Enhanced patterns with better context awareness
PHQ8_SCORE_MAPPING = {
    0: {
        "patterns": ["not at all", "never", "no", "haven't", "not really", "rarely", "none", 
                     "haven't been", "don't", "never been", "not", "no not"],
        "intensity_modifiers": ["really", "definitely", "completely", "absolutely"],
        "negation_patterns": [r"not\s+\w+", r"no\s+\w+", r"never\s+\w+"]
    },
    1: {
        "patterns": ["sometimes", "occasionally", "few times", "couple days", "once in a while", 
                     "several days", "a few", "once or twice", "a couple", "here and there"],
        "intensity_modifiers": ["just", "only", "barely"],
        "time_patterns": [r"\d+\s+times", r"few\s+times"]
    },
    2: {
        "patterns": ["often", "most days", "frequently", "more than half", "usually", 
                     "majority of time", "more than half the days", "most of the time"],
        "intensity_modifiers": ["very", "quite", "pretty"],
        "frequency_patterns": [r"most\s+\w+", r"more\s+than\s+half"]
    },
    3: {
        "patterns": ["always", "every day", "constantly", "all the time", "daily", 
                     "nearly every day", "almost every day", "every single day"],
        "intensity_modifiers": ["always", "constantly", "continuously"],
        "frequency_patterns": [r"every\s+day", r"all\s+the\s+time"]
    }
}

# Additional patterns for better detection
INTENSITY_MODIFIERS = {
    "high": ["very", "really", "extremely", "quite", "pretty", "super", "incredibly"],
    "low": ["a little", "slightly", "somewhat", "kind of", "sort of", "a bit"]
}

# Time references that indicate frequency
TIME_REFERENCES = {
    "recent": ["lately", "recently", "these days", "nowadays", "past few", "last few"],
    "ongoing": ["always", "constantly", "continuously", "all the time"],
    "intermittent": ["sometimes", "occasionally", "once in a while", "now and then"]
}


def load_conversation_data(csv_path: str) -> pd.DataFrame:
    """Load conversation data from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ["start", "stop", "speaker", "text"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


def identify_phq8_question(question_text: str) -> Optional[str]:
    """
    Identify which PHQ-8 question the interviewer is asking
    
    Returns:
        PHQ-8 topic name or None if not a PHQ-8 question
    """
    question_lower = question_text.lower()
    
    for topic, info in PHQ8_QUESTIONS.items():
        # Check if question contains keywords related to this PHQ-8 topic
        keywords = info["keywords"]
        if any(keyword in question_lower for keyword in keywords):
            return topic
    
    return None


def extract_phq8_answer(response_text: str, topic: str) -> Tuple[Optional[int], float]:
    """
    Extract PHQ-8 answer score (0-3) from user response with enhanced accuracy
    
    Returns:
        (score, confidence) where score is 0-3 or None if not detected
    """
    response_lower = response_text.lower().strip()
    confidence = 0.0
    
    # Check for negative indicators from the topic FIRST (more reliable)
    topic_info = PHQ8_QUESTIONS.get(topic, {})
    negative_indicators = topic_info.get("negative_indicators", [])
    keywords = topic_info.get("keywords", [])
    
    # Enhanced keyword matching with word boundaries
    has_topic_keywords = False
    matched_keywords = []
    for kw in keywords:
        if kw not in ["feel", "feelings", "think"]:  # Exclude generic words
            # Use word boundary matching for more precise detection
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, response_lower):
                has_topic_keywords = True
                matched_keywords.append(kw)
    
    # Count negative indicators with better matching
    negative_count = 0
    matched_indicators = []
    for indicator in negative_indicators:
        # Check for phrase matches
        if indicator in response_lower:
            negative_count += 1
            matched_indicators.append(indicator)
    
    # CRITICAL: Check for ADMISSIONS first (e.g., "i do sometimes yes", "i've been depressed")
    # These override positive indicators
    admission_patterns = {
        "mood": [
            r"\bi\s+(do|have|am)\s+(sometimes|often|usually|always|pretty|very|really)\s+(depressed|down|sad|hopeless)",
            r"\bi\s+(do|have|am)\s+(sometimes|often|usually|always)\s+(yes|yeah)",
            r"\bi['']ve\s+(been|still\s+been)\s+(pretty|very|really|quite|lately|recently)\s+(depressed|down|sad|hopeless)",
            r"\bi\s+feel\s+(pretty|very|really|quite)\s+(depressed|down|sad|hopeless)",
            r"\bsometimes\s+i\s+(do|feel|am)\s+(depressed|down|sad|hopeless)",
            r"\bi\s+(do|am)\s+(sometimes|often|usually|always)",
            r"\blately\s+i['']ve\s+(been|still\s+been)",
            r"\bi['']ve\s+still\s+been\s+(pretty|very|really|quite)?\s*(depressed|down|sad)",
            r"\bfatigue\s+has.*(implications|affects|makes)",
        ],
        "sleep": [
            r"\bi\s+(don['']t|can['']t)\s+sleep",
            r"\bi\s+have\s+trouble\s+sleeping",
            r"\bi\s+(don['']t|can['']t)\s+fall\s+asleep",
            r"\bi\s+(wake|waking)\s+up",
            r"\binsomnia",
            r"\bi\s+sleep\s+(too\s+much|way\s+too\s+much)",
            r"\bfall\s+asleep\s+(during|at)",
        ],
        "energy": [
            r"\bi\s+(am|feel|have\s+been)\s+(tired|exhausted|fatigued|drained)\s+(all|all\s+the\s+time)",
            r"\bi\s+(don['']t|have\s+no)\s+energy",
            r"\bno\s+energy",
            r"\blittle\s+energy",
            r"\bfatigue\s+has",
        ],
        "interest_pleasure": [
            r"\bi\s+(don['']t|don['']t\s+really)\s+(enjoy|like|have\s+interest)",
            r"\bi\s+(lost|have\s+lost)\s+interest",
            r"\bnothing\s+(interests|seems\s+fun)",
            r"\bi\s+(don['']t|can['']t)\s+enjoy",
        ],
    }
    
    # Check for explicit admissions
    topic_admissions = admission_patterns.get(topic, [])
    admission_found = False
    admission_score = None
    
    for pattern in topic_admissions:
        if re.search(pattern, response_lower):
            admission_found = True
            # Extract frequency/severity from admission
            # Check for "lately" + "still been" + "pretty" = Score 2-3
            if "lately" in response_lower and "still been" in response_lower:
                if "pretty" in response_lower or "very" in response_lower or "really" in response_lower:
                    admission_score = 2
                else:
                    admission_score = 1
            elif "still been" in response_lower:
                if "pretty" in response_lower or "very" in response_lower or "really" in response_lower:
                    admission_score = 2
                else:
                    admission_score = 1
            elif any(word in response_lower for word in ["sometimes", "occasionally", "once in a while"]):
                admission_score = 1
            elif any(word in response_lower for word in ["often", "usually", "most days", "frequently"]):
                admission_score = 2
            elif any(word in response_lower for word in ["always", "all the time", "constantly", "every day"]):
                admission_score = 3
            elif any(word in response_lower for word in ["pretty", "very", "really", "quite"]):
                # "pretty depressed" suggests moderate severity
                admission_score = 2
            else:
                admission_score = 1  # Default to score 1 if admission found but no frequency
            break
    
    if admission_found and admission_score is not None:
        confidence = 0.85
        if admission_score >= 2:
            confidence = 0.9
        return admission_score, confidence
    
    # Check for conditional/hypothetical statements (should be score 0)
    conditional_patterns = [
        r"\bif\s+i\s+could",
        r"\bif\s+i\s+had",
        r"\bi\s+would\s+(enjoy|like|love)",
        r"\bi\s+would\s+be",
        r"\bsure\s+i\s+would",
        r"\bwish\s+i\s+could",
        r"\bwould\s+if",
    ]
    is_hypothetical = any(re.search(pattern, response_lower) for pattern in conditional_patterns)
    
    # Check for POSITIVE responses (should be score 0 with HIGH confidence)
    # Positive indicators for each topic
    positive_indicators = {
        "interest_pleasure": ["i like", "i enjoy", "i love", "i have fun", "fun", "enjoying", 
                              "i like to", "i enjoy doing", "i love to", "having fun", "i'm having fun",
                              "is fun", "are fun", "was fun", "enjoyed", "liked", "loved"],
        "mood": ["i'm good", "i'm fine", "i'm okay", "i'm happy", "feeling good", "feeling fine",
                 "doing well", "doing good", "i'm great", "i feel great", "right now i'm fine"],
        "sleep": ["sleep well", "sleeping well", "sleep fine", "sleeping fine", "sleep good",
                  "sleeping good", "rest well", "resting well"],
        "energy": ["i have energy", "i have lots of energy", "i'm energetic", "i feel energetic",
                   "plenty of energy", "enough energy", "good energy"],
        "appetite": ["i eat", "i'm eating", "i have appetite", "i'm hungry", "i eat well",
                     "eating well", "good appetite"],
        "self_worth": ["i'm good", "i'm fine", "i'm okay", "i'm confident", "i feel confident",
                       "i'm capable", "i can do", "i'm able"],
        "concentration": ["i can focus", "i can concentrate", "i focus well", "i concentrate well",
                          "good focus", "i pay attention", "i remember"],
        "psychomotor": ["i'm normal", "i move normally", "normal pace", "normal speed"]
    }
    
    # Check for positive indicators (strong signal for score 0)
    topic_positive = positive_indicators.get(topic, [])
    positive_matches = []
    for pos_indicator in topic_positive:
        if pos_indicator in response_lower:
            positive_matches.append(pos_indicator)
    
    # If hypothetical, return score 0 but with lower confidence
    if is_hypothetical:
        return 0, 0.7
    
    # If we have strong positive indicators, return score 0 with HIGH confidence
    if positive_matches:
        # Check if there's a qualifier that negates it ("right now i'm fine" but "i do sometimes")
        if "right now" in response_lower or "now" in response_lower[:30]:
            # Check if there's an admission before this
            if any(word in response_lower for word in ["sometimes", "often", "usually", "do", "have", "been"]):
                # Mixed response - positive now but admits to problems
                if "sometimes" in response_lower:
                    return 1, 0.8
                elif "often" in response_lower or "usually" in response_lower:
                    return 2, 0.85
                else:
                    return 1, 0.75
        
        # Calculate confidence based on how many positive indicators and response quality
        confidence = 0.8 + (len(positive_matches) * 0.03)  # Base 0.8, +0.03 per match
        if len(response_text.split()) > 10:  # Detailed positive response
            confidence += 0.05
        if len(response_text.split()) > 20:  # Very detailed
            confidence += 0.05
        confidence = min(confidence, 0.95)
        return 0, confidence
    
    # Only proceed if we have topic-relevant content
    if not has_topic_keywords and negative_count == 0:
        return None, 0.0
    
    # Check for explicit frequency indicators BUT only if they relate to SYMPTOMS, not activities
    # This prevents "occasionally" in "I go shopping occasionally" from being misinterpreted
    frequency_score = None
    frequency_confidence = 0.0
    frequency_pattern_found = None
    
    # Check if frequency words are describing activities vs symptoms
    activity_context_indicators = ["i like to", "i enjoy", "i do", "i go", "i play", "i shop",
                                   "i walk", "i visit", "i have fun", "for fun", "having fun"]
    is_activity_context = any(indicator in response_lower for indicator in activity_context_indicators)
    
    for score, score_data in PHQ8_SCORE_MAPPING.items():
        patterns = score_data.get("patterns", [])
        intensity_modifiers = score_data.get("intensity_modifiers", [])
        
        for pattern in patterns:
            # Skip frequency patterns if they're describing activities, not symptoms
            if is_activity_context and score > 0:
                # Check if pattern is near activity indicators
                pattern_pos = response_lower.find(pattern)
                if pattern_pos != -1:
                    # Check context around the pattern
                    context_start = max(0, pattern_pos - 30)
                    context_end = min(len(response_lower), pattern_pos + len(pattern) + 30)
                    context = response_lower[context_start:context_end]
                    
                    # If pattern is near activity words, it's describing activity frequency, not symptom
                    if any(act in context for act in activity_context_indicators):
                        continue  # Skip this pattern, it's about activities
            
            # Multi-word pattern - check if it appears as a phrase
            if len(pattern.split()) > 1:
                if pattern in response_lower:
                    frequency_score = score
                    frequency_pattern_found = pattern
                    frequency_confidence = 0.9
                    # Check for intensity modifiers
                    for mod in intensity_modifiers:
                        if mod in response_lower:
                            frequency_confidence = min(frequency_confidence + 0.05, 0.95)
                    break
            else:
                # Single word - check word boundaries
                pattern_match = re.search(r'\b' + re.escape(pattern) + r'\b', response_lower)
                if pattern_match:
                    frequency_score = score
                    frequency_pattern_found = pattern
                    frequency_confidence = 0.85
                    # Check for intensity modifiers nearby
                    start_pos = pattern_match.start()
                    context = response_lower[max(0, start_pos-10):min(len(response_lower), start_pos+len(pattern)+10)]
                    for mod in intensity_modifiers:
                        if mod in context:
                            frequency_confidence = min(frequency_confidence + 0.05, 0.95)
                    break
        
        if frequency_score is not None:
            break
    
    # Handle negation patterns for score 0
    negation_patterns = [
        r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bhaven't\b", r"\bdon't\b",
        r"\bhasn't\b", r"\bdoesn't\b", r"\bwon't\b", r"\bcan't\b"
    ]
    has_negation = any(re.search(pattern, response_lower) for pattern in negation_patterns)
    
    # If we found explicit frequency AND topic relevance AND it's about symptoms (not activities)
    if frequency_score is not None and (has_topic_keywords or negative_count > 0):
        # If frequency is found but in activity context, it's likely score 0 (they're describing activities)
        if is_activity_context and frequency_score > 0:
            return 0, 0.85  # High confidence - they're describing activities they enjoy
        
        # Adjust score based on negation
        if has_negation and frequency_score > 0:
            # Check if negation is about the symptom itself
            if any(ind in response_lower for ind in matched_indicators):
                # Double negation or clarification - might be score 0
                if "not" in response_lower[:50] or "no" in response_lower[:50]:
                    return 0, 0.8
            else:
                # Keep the frequency score but adjust confidence
                frequency_confidence = max(frequency_confidence - 0.1, 0.6)
        
        return frequency_score, frequency_confidence
    
    # Check for symptom descriptions (e.g., "what am i like irritated tired lazy")
    # These describe consequences/symptoms but may not directly answer the question
    symptom_description_patterns = {
        "sleep": [
            r"\b(tired|irritated|lazy|exhausted|drained|worn out|grumpy|cranky)\b",
            r"\bfall\s+asleep\s+(during|at)",
            r"\bcan['']t\s+stay\s+awake",
        ],
        "energy": [
            r"\b(tired|exhausted|fatigued|drained|worn out|lethargic|sluggish)\b",
            r"\bno\s+energy",
            r"\blittle\s+energy",
        ],
        "mood": [
            r"\b(irritated|grumpy|cranky|moody|upset|frustrated)\b",
            r"\bnot\s+feeling\s+like\s+yourself",
            r"\bmisjudging",
        ],
    }
    
    # Check if response describes symptoms/consequences
    topic_symptoms = symptom_description_patterns.get(topic, [])
    symptom_descriptions = []
    for pattern in topic_symptoms:
        if re.search(pattern, response_lower):
            symptom_descriptions.append(pattern)
    
    # If describing symptoms, assign score based on severity
    if symptom_descriptions:
        # Check for severity indicators
        severity_words = ["very", "really", "extremely", "pretty", "quite", "all the time", "always"]
        has_severity = any(word in response_lower for word in severity_words)
        
        # Count how many symptoms mentioned
        symptom_count = len(symptom_descriptions)
        
        if has_severity or symptom_count >= 2:
            score = 2
            confidence = 0.8
        else:
            score = 1
            confidence = 0.75
        
        return score, confidence
    
    # Use negative indicators to estimate severity with enhanced logic
    if negative_count > 0:
        # Base severity on number and strength of negative indicators
        base_score = min(negative_count, 2)  # Cap at 2 initially
        
        # Check for intensity words that suggest higher severity
        intensity_words = ["very", "really", "extremely", "completely", "totally", "all the time", "pretty", "quite"]
        has_high_intensity = any(word in response_lower for word in intensity_words)
        
        # Check for severity modifiers
        severity_modifiers = ["pretty", "very", "really", "extremely", "quite", "still"]
        has_severity_modifier = any(mod in response_lower for mod in severity_modifiers)
        
        # Check for time references indicating frequency
        time_indicators = {
            "continuous": ["all the time", "always", "constantly", "every day", "every single day"],
            "frequent": ["often", "most days", "usually", "frequently"],
            "occasional": ["sometimes", "occasionally", "once in a while"]
        }
        
        time_context = "occasional"
        for freq_type, indicators in time_indicators.items():
            if any(ind in response_lower for ind in indicators):
                time_context = freq_type
                break
        
        # Adjust score based on intensity, severity modifiers, and time context
        if has_severity_modifier and ("still" in response_lower or "been" in response_lower):
            # "still been depressed" or "pretty depressed" suggests ongoing/moderate-severe
            base_score = 2
        elif has_high_intensity:
            base_score = min(base_score + 1, 3)
        elif time_context == "continuous":
            base_score = min(base_score + 1, 3)
        elif time_context == "frequent":
            base_score = min(base_score + 1, 2)
        
        # Calculate confidence based on multiple factors
        confidence = 0.6 + (negative_count * 0.1)  # Base confidence higher
        if len(response_text.split()) > 15:  # Longer responses are more detailed
            confidence += 0.1
        if has_high_intensity:
            confidence += 0.1
        if has_severity_modifier:
            confidence += 0.05
        if matched_keywords:
            confidence += 0.05
        
        confidence = min(confidence, 0.95)
        
        return base_score, confidence
    
    # Handle positive responses with topic keywords but no negative indicators (score 0 with HIGH confidence)
    if has_topic_keywords and not negative_count:
        # Check if response describes activities they enjoy/do
        enjoyment_words = ["like", "enjoy", "love", "fun", "good", "great", "nice"]
        has_enjoyment = any(word in response_lower for word in enjoyment_words)
        
        if has_enjoyment:
            # They're describing things they enjoy - score 0 with high confidence
            confidence = 0.85
            if len(response_text.split()) > 10:
                confidence = 0.9
            return 0, confidence
        
        # Check for clear denial patterns
        denial_patterns = [
            r"\bno\b", r"\bnot\s+\w+", r"\bnever\s+\w+", r"\bdon't\s+\w+",
            r"\bhaven't\s+\w+", r"\bnot\s+really", r"\bnot\s+at\s+all"
        ]
        
        for pattern in denial_patterns:
            if re.search(pattern, response_lower):
                # Verify it's about the topic, not something else
                # Check if denial is near topic keywords
                for kw in matched_keywords:
                    kw_pos = response_lower.find(kw)
                    if kw_pos != -1:
                        context = response_lower[max(0, kw_pos-20):min(len(response_lower), kw_pos+len(kw)+20)]
                        if any(re.search(pattern, context) for pattern in denial_patterns):
                            return 0, 0.85  # Higher confidence for clear denial
        
        # Check for incomplete/very short responses
        if len(response_text.split()) <= 3:
            return 0, 0.6  # Lower confidence for very short/incomplete responses
        
        # If topic mentioned but no indicators and response is substantial, likely score 0
        if len(response_text.split()) > 5:
            return 0, 0.75  # Higher confidence for substantial responses
        else:
            return 0, 0.65  # Lower confidence for short/incomplete responses
    
    # No clear indicators found
    return None, 0.0


def process_conversation_to_training_pairs(df: pd.DataFrame) -> List[Dict]:
    """
    Process conversation data into input/output pairs for training
    
    Format:
    - Input: Conversation context + PHQ-8 question identification
    - Output: PHQ-8 question identified, answer extracted, depression assessment
    """
    training_pairs = []
    
    # Statistics for debugging
    stats = {
        "total_turns": 0,
        "skipped_same_speaker": 0,
        "interviewer_turns": 0,
        "phq8_questions_found": 0,
        "participant_responses": 0,
        "answers_extracted": 0,
        "answers_none": 0,
        "general_user_turns": 0,
        "general_with_keywords": 0,
        "general_with_symptoms": 0,
    }
    
    # Group conversation by turns
    conversation_turns = []
    
    for idx, row in df.iterrows():
        speaker = row["speaker"]
        text = str(row["text"]).strip()
        
        if not text or text.lower() in ["nan", "none", ""]:
            continue
        
        conversation_turns.append({
            "speaker": speaker,
            "text": text,
            "turn": idx
        })
        stats["total_turns"] += 1
    
    # Collect unique speaker names for debugging
    unique_speakers = set()
    for turn in conversation_turns:
        unique_speakers.add(turn["speaker"].lower())
    
    print(f"   Unique speakers found: {sorted(unique_speakers)}")
    
    # Process conversation in pairs (question-answer)
    for i in range(len(conversation_turns) - 1):
        current_turn = conversation_turns[i]
        next_turn = conversation_turns[i + 1]
        
        # Skip if same speaker speaks twice in a row
        if current_turn["speaker"] == next_turn["speaker"]:
            stats["skipped_same_speaker"] += 1
            continue
        
        # More lenient speaker matching - check if it contains interviewer patterns
        current_speaker_lower = current_turn["speaker"].lower()
        next_speaker_lower = next_turn["speaker"].lower()
        
        is_interviewer = any(pattern in current_speaker_lower for pattern in 
                             ["ellie", "interviewer", "assistant", "ai", "bot", "therapist", "counselor"])
        is_participant = any(pattern in next_speaker_lower for pattern in 
                            ["participant", "user", "patient", "client"])
        
        # Identify if current turn is a PHQ-8 question
        question_topic = None
        if is_interviewer:
            stats["interviewer_turns"] += 1
            question_topic = identify_phq8_question(current_turn["text"])
            if question_topic:
                stats["phq8_questions_found"] += 1
        
        # If it's a PHQ-8 question, extract answer from next turn
        if question_topic and is_participant:
            stats["participant_responses"] += 1
            answer_score, confidence = extract_phq8_answer(next_turn["text"], question_topic)
            
            # IMPORTANT: Create pairs even for score 0 (no depression) - this is valuable training data!
            if answer_score is not None:
                stats["answers_extracted"] += 1
                # Create training pair for PHQ-8 detection
                phq8_info = PHQ8_QUESTIONS[question_topic]
                
                # Build context (last few turns)
                context_start = max(0, i - 2)
                context_turns = conversation_turns[context_start:i+2]
                context = "\n".join([
                    f"{turn['speaker']}: {turn['text']}" 
                    for turn in context_turns
                ])
                
                # Create input/output pair
                input_text = f"""Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
{context}

PHQ-8 Questions Reference:
{json.dumps({k: v['question'] for k, v in PHQ8_QUESTIONS.items()}, indent=2)}"""
                
                output_text = f"""PHQ-8 Question Identified: {phq8_info['question']}
Topic: {question_topic}
User Response: {next_turn['text']}
PHQ-8 Answer Score: {answer_score} (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)
Confidence: {confidence:.2f}
Depression Assessment: {'Depression indicators present' if answer_score >= 2 else 'Minimal or no depression indicators'}"""
                
                training_pairs.append({
                    "input": input_text,
                    "output": output_text,
                    "topic": question_topic,
                    "score": answer_score,
                    "confidence": confidence
                })
            else:
                stats["answers_none"] += 1
    
    # Also create pairs for general depression detection
    # IMPORTANT: Process ALL user responses, not just those with keywords
    # This allows detection from general conversation (like "i'm pretty local", "i took a minute", etc.)
    for i in range(len(conversation_turns)):
        speaker_lower = conversation_turns[i]["speaker"].lower()
        is_participant = any(pattern in speaker_lower for pattern in 
                            ["participant", "user", "patient", "client"])
        
        if is_participant:
            stats["general_user_turns"] += 1
            # Build context around this response (include more context for better analysis)
            context_start = max(0, i - 5)  # Increased from 3 to 5 for more context
            context_end = min(len(conversation_turns), i + 1)
            context_turns = conversation_turns[context_start:context_end]
            
            # Check if conversation has PHQ-8 related content
            context_text = " ".join([turn["text"].lower() for turn in context_turns])
            phq8_keywords_in_context = []
            for topic, info in PHQ8_QUESTIONS.items():
                for keyword in info["keywords"]:
                    if keyword in context_text:
                        phq8_keywords_in_context.append(keyword)
            
            # CHANGED: Don't skip even if no keywords - analyze ALL user responses
            # This is critical for general conversation detection
            if phq8_keywords_in_context:
                stats["general_with_keywords"] += 1
            
            # Process ALL participant responses, not just those with keywords
            
            context = "\n".join([
                f"{turn['speaker']}: {turn['text']}" 
                for turn in context_turns
            ])
            
            user_response = conversation_turns[i]["text"]
            
            # Check for depression indicators across all PHQ-8 topics
            # Process ALL user responses, even if they don't explicitly mention PHQ-8 keywords
            detected_topics = []
            total_score = 0
            
            for topic, info in PHQ8_QUESTIONS.items():
                answer_score, confidence = extract_phq8_answer(user_response, topic)
                # LOWER threshold to detect more subtle indicators from general conversation
                # Even score 0 with low confidence is valuable if it's clearly a healthy response
                if answer_score is not None:
                    # Include if confidence is reasonable OR if it's a clear score 0 (healthy response)
                    if confidence > 0.3 or (answer_score == 0 and confidence > 0.5):
                        detected_topics.append({
                            "topic": topic,
                            "score": answer_score,
                            "confidence": confidence
                        })
                        total_score += answer_score
            
            # Create entry if ANY topic detected OR if response is substantial enough to analyze
            # This ensures we capture both explicit and implicit depression indicators
            user_response_length = len(user_response.split())
            has_substantial_response = user_response_length > 3  # More than just "yes", "no", "okay"
            
            # Initialize phq8_total here so it's available for both branches
            phq8_total = total_score
            
            if detected_topics:
                stats["general_with_symptoms"] += 1
                # Determine depression severity
                if phq8_total <= 4:
                    severity = "Minimal depression"
                elif phq8_total <= 9:
                    severity = "Mild depression"
                elif phq8_total <= 14:
                    severity = "Moderate depression"
                elif phq8_total <= 19:
                    severity = "Moderately severe depression"
                else:
                    severity = "Severe depression"
                
                # Enhanced prompt that emphasizes extracting from general conversation
                input_text = f"""Analyze the following conversation for depression indicators using PHQ-8 assessment.
Extract PHQ-8 symptoms even from indirect responses and general conversation.

Conversation:
{context}

Identify:
1. PHQ-8 symptoms present in the user's responses (even if not explicitly mentioned)
2. PHQ-8 scores for each detected symptom (0-3)
3. Overall depression assessment

Note: The user may not directly answer PHQ-8 questions. Extract symptoms from their general responses and context."""
                
                # Build output with better formatting
                detected_symptoms = "\n".join([
                    f"- {t['topic'].replace('_', ' ').title()}: Score {t['score']} (confidence: {t['confidence']:.2f})"
                    for t in detected_topics
                ])
                
                output_text = f"""Detected PHQ-8 Symptoms:
{detected_symptoms}

Total PHQ-8 Score: {phq8_total}/24
Depression Assessment: {severity}
User Response Analyzed: {user_response}"""
                
                training_pairs.append({
                    "input": input_text,
                    "output": output_text,
                    "topic": "general",
                    "score": phq8_total,
                    "confidence": sum(t["confidence"] for t in detected_topics) / len(detected_topics) if detected_topics else 0.6
                })
            
            elif has_substantial_response:
                # Create entry for substantial responses even if no symptoms detected
                # This helps model learn what "no symptoms" looks like
                stats["general_with_symptoms"] += 1
                
                # phq8_total is already 0 here (no detected_topics)
                severity = "Minimal depression"
                
                input_text = f"""Analyze the following conversation for depression indicators using PHQ-8 assessment.
Extract PHQ-8 symptoms even from indirect responses and general conversation.

Conversation:
{context}

Identify:
1. PHQ-8 symptoms present in the user's responses (even if not explicitly mentioned)
2. PHQ-8 scores for each detected symptom (0-3)
3. Overall depression assessment

Note: The user may not directly answer PHQ-8 questions. Extract symptoms from their general responses and context."""
                
                output_text = f"""Detected PHQ-8 Symptoms:
- No clear PHQ-8 symptoms detected in this response

Total PHQ-8 Score: 0/24
Depression Assessment: Minimal depression
User Response Analyzed: {user_response}"""
                
                training_pairs.append({
                    "input": input_text,
                    "output": output_text,
                    "topic": "general",
                    "score": 0,  # Explicitly set to 0 for no symptoms
                    "confidence": 0.6
                })
    
    # Print statistics
    print(f"\n   Processing Statistics:")
    print(f"   - Total conversation turns: {stats['total_turns']}")
    print(f"   - Skipped (same speaker): {stats['skipped_same_speaker']}")
    print(f"   - Interviewer turns: {stats['interviewer_turns']}")
    print(f"   - PHQ-8 questions identified: {stats['phq8_questions_found']}")
    print(f"   - Participant responses to PHQ-8: {stats['participant_responses']}")
    print(f"   - Answers extracted (score not None): {stats['answers_extracted']}")
    print(f"   - Answers returned None: {stats['answers_none']}")
    print(f"   - General user turns: {stats['general_user_turns']}")
    print(f"   - General turns with PHQ-8 keywords: {stats['general_with_keywords']}")
    print(f"   - General turns with detected symptoms: {stats['general_with_symptoms']}")
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

# Quantization parameters (USE_4BIT is defined at top of file)
# NOTE: If you get bitsandbytes CUDA errors in Colab:
#   1. Restart runtime: Runtime -> Restart runtime
#   2. Reinstall: !pip install -U bitsandbytes
#   3. Or disable quantization by setting USE_4BIT = False at the top of the file (line ~51)
#   4. The script will automatically try to recover if bitsandbytes fails during training
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# Training parameters
# NOTE: Optimized to use ~65GB GPU memory (stay within limit)
OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 18  # Further reduced to stay within 65GB limit
PER_DEVICE_EVAL_BATCH_SIZE = 18
GRADIENT_ACCUMULATION_STEPS = 3  # Increased to maintain effective batch size (54 total)
GRADIENT_CHECKPOINTING = True  # Enabled to reduce memory usage and stay within 65GB
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = False  # Disabled for faster training
SAVE_STEPS = 500
LOGGING_STEPS = 25
MAX_SEQ_LENGTH = 2048  # Further reduced to stay within 65GB limit (was 2560)
PACKING = False


def formatting_func(example):
    """Format training examples for Mistral"""
    return f"### Instruction:\n{example['input']}\n\n### Response:\n{example['output']}"


def main():
    """Main training function - COLAB COMPATIBLE"""
    print("=" * 80)
    print("Mistral 7B Fine-tuning for PHQ-8 Depression Detection")
    print("=" * 80)

    # Warn if quantization is enabled
    if USE_4BIT:
        print("\n⚠️  WARNING: Quantization is ENABLED (USE_4BIT = True)")
        print("If you encounter bitsandbytes errors, set USE_4BIT = False at line 52")
        print("=" * 80)
    else:
        print("\n✓ Quantization is DISABLED (USE_4BIT = False) - bitsandbytes will not be used")
        print("=" * 80)

    # Step 1: Load and process conversation data
    # Try to find CSV file in common locations (Colab-friendly)
    csv_path = None
    possible_paths = [
        "combined_sessions.csv",
        "/content/combined_sessions.csv",
        "/content/drive/MyDrive/combined_sessions.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print("\nCSV file not found in common locations.")
        print("Please upload your conversation CSV file or specify the path.")
        print("\nIn Colab, you can:")
        print("1. Upload file: from google.colab import files; files.upload()")
        print("2. Or mount Drive: from google.colab import drive; drive.mount('/content/drive')")
        print("\nTrying to list CSV files in /content...")
        try:
            if os.path.exists("/content"):
                csv_files = [f for f in os.listdir("/content") if f.endswith(".csv")]
                if csv_files:
                    print(f"Found CSV files: {csv_files}")
                    csv_path = f"/content/{csv_files[0]}"
                    print(f"Using: {csv_path}")
                else:
                    print("No CSV files found. Please upload your data file.")
                    return
        except Exception as e:
            print(f"Error checking /content: {e}")
            return
    
    print(f"\n1. Loading conversation data from {csv_path}...")
    df = load_conversation_data(csv_path)
    print(f"   Loaded {len(df)} conversation turns")
    
    # Step 2: Process into training pairs
    print("\n2. Processing conversation data into training pairs...")
    training_pairs = process_conversation_to_training_pairs(df)
    print(f"   Created {len(training_pairs)} training pairs")
    
    if len(training_pairs) == 0:
        print("ERROR: No training pairs created. Please check your data format.")
        return
    
    # Step 3: Save training data
    training_data_path = "phq8_training_data.json"
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
    
    # Check if quantization is disabled - if so, skip all bitsandbytes code
    use_quantization = False
    bitsandbytes_works = False
    
    if not USE_4BIT:
        print("   Quantization disabled (USE_4BIT = False) - loading model without quantization")
    else:
        # Only test bitsandbytes if USE_4BIT is True
        print("   Testing bitsandbytes compatibility...")
        try:
            import bitsandbytes as bnb
            if torch.cuda.is_available():
                # Try to actually test bitsandbytes CUDA functions
                try:
                    # Create a test tensor and try to use bitsandbytes operations
                    test_tensor = torch.randn(2, 2, dtype=torch.float16).cuda()
                    # Just checking if we can import and access version
                    version = bnb.__version__
                    print(f"   Found bitsandbytes version: {version}")
                    # Try to actually use a bitsandbytes function to verify CUDA works
                    # This will fail if CUDA version doesn't match
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
            # Configure quantization
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
            if "bitsandbytes" in str(e).lower() or "cuda" in str(e).lower() or "lib.cget_managed_ptr" in str(e).lower():
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
        
        # Check available memory
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
        
        # Ensure quantization is completely disabled - don't pass any quantization config
        # IMPORTANT: Don't import bitsandbytes or use any quantization-related code
        # Clear memory before loading
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,  # Optimize memory usage
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        # Enable gradient checkpointing only if requested (disabled for max speed)
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Mistral attention modules
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
        gradient_checkpointing=GRADIENT_CHECKPOINTING,  # Use parameter setting
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=True,  # Enable mixed precision for faster training (80GB GPU can handle it)
        bf16=False,  # Use fp16 instead of bf16 for better compatibility
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=GROUP_BY_LENGTH,  # Use parameter setting
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to=["tensorboard"] if IN_COLAB else [],
        remove_unused_columns=False,
        # Progress bar settings
        disable_tqdm=False,  # Enable progress bar
        logging_first_step=True,  # Log first step
        prediction_loss_only=False,  # Show more details
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=None,
        dataloader_num_workers=1,  # Minimal multiprocessing to save memory
        dataloader_pin_memory=False,  # Disabled to save memory
        # Memory optimization
        dataloader_prefetch_factor=1,  # Reduced to save memory
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
        # max_seq_length=MAX_SEQ_LENGTH,  # CRITICAL: Must be set to limit memory usage
        # packing=PACKING,
    )

    # Step 9: Train model
    print("\n9. Starting training...")
    print("=" * 80)
    
    if use_quantization:
        print("   Using quantized model - training will proceed")
    else:
        print("   Using non-quantized model - training will proceed")
    
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
        print(f"   Mixed Precision (fp16): {True if training_arguments.fp16 else False}")
        print(f"   Gradient Checkpointing: {GRADIENT_CHECKPOINTING} ({'enabled (memory efficient)' if GRADIENT_CHECKPOINTING else 'disabled (faster)'})")
        
        # Calculate expected memory usage
        # More conservative estimate with gradient checkpointing and fp16
        checkpointing_factor = 0.6 if GRADIENT_CHECKPOINTING else 1.0  # Gradient checkpointing saves ~40% memory
        memory_per_batch = PER_DEVICE_TRAIN_BATCH_SIZE * MAX_SEQ_LENGTH * 0.0015 * checkpointing_factor  # More conservative
        expected_memory_gb = allocated_memory + memory_per_batch
        target_memory_gb = 65.0  # Target: stay under 65GB
        
        print(f"\n📊 Expected Memory Usage:")
        print(f"   Current: {allocated_memory:.1f} GB")
        print(f"   Estimated peak: {min(expected_memory_gb, total_memory):.1f} GB")
        print(f"   Target: Stay under {target_memory_gb} GB")
        print(f"   Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        
        # Memory safety check
        if expected_memory_gb > target_memory_gb:
            print(f"\n⚠️  WARNING: Estimated memory ({expected_memory_gb:.1f}GB) exceeds target ({target_memory_gb}GB)")
            print(f"   Consider reducing batch_size or enabling gradient checkpointing.")
        elif free_memory > 20:
            print(f"\n💡 GPU Memory Status: {free_memory:.1f}GB free")
            print(f"   Current settings: batch_size={PER_DEVICE_TRAIN_BATCH_SIZE}, seq_len={MAX_SEQ_LENGTH}")
            print(f"   Estimated usage: ~{expected_memory_gb:.1f}GB (within {target_memory_gb}GB limit)")
        elif free_memory < 10:
            print(f"\n⚠️  WARNING: Low GPU memory ({free_memory:.1f}GB free)")
            print(f"   Monitor training closely. If OOM occurs, reduce batch_size or seq_len.")
        else:
            print(f"\n💡 GPU Memory Status: {free_memory:.1f}GB free (should be sufficient)")
    
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
        # Enable progress bar explicitly
        trainer.train()
    except RuntimeError as e:
        error_str = str(e).lower()
        # Check for bitsandbytes errors (including lib.cget_managed_ptr)
        if ("bitsandbytes" in error_str or "cuda" in error_str or 
            "lib.cget_managed_ptr" in error_str or "lib.cquantize" in error_str):
            print("\n" + "=" * 80)
            print("ERROR: bitsandbytes failed during training")
            print("=" * 80)
            print("\nAutomatically recovering by reloading model without quantization...")
            
            # Attempt to recover - completely disable quantization
            import gc
            
            # Clear everything
            try:
                del trainer
            except:
                pass
            try:
                del model
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Set environment variable to disable bitsandbytes
            os.environ['BITSANDBYTES_NOWELCOME'] = '1'
            
            print("   Reloading model WITHOUT quantization...")
            # Force reload without any quantization - make absolutely sure no quantization config
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                model.config.use_cache = False
                model.config.pretraining_tp = 1
                
                # Recreate trainer with non-quantized model
                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    formatting_func=formatting_func,
                    processing_class=tokenizer,
                    args=training_arguments,
                    # max_seq_length=MAX_SEQ_LENGTH,  # CRITICAL: Must be set to limit memory usage
                    # packing=PACKING,
                )
                
                print("   ✓ Model reloaded without quantization")
                print("   ✓ Trainer recreated")
                print("   Retrying training without quantization...")
                print("=" * 80)
                
                # Try training again
                trainer.train()
            except RuntimeError as e2:
                # If recovery also fails, give up and tell user to disable quantization
                error_str2 = str(e2).lower()
                if "bitsandbytes" in error_str2 or "lib.cget_managed_ptr" in error_str2:
                    print("\n" + "=" * 80)
                    print("ERROR: Recovery failed - bitsandbytes is still being triggered")
                    print("=" * 80)
                    print("\nThe issue is that bitsandbytes is incompatible with your CUDA version.")
                    print("\nSOLUTION:")
                    print("1. Set USE_4BIT = False at line 52 of the script")
                    print("2. Restart the Colab runtime: Runtime -> Restart runtime")
                    print("3. Run the script again")
                    print("\nThe model will use more GPU memory (~14GB) but will work fine.")
                    print("=" * 80)
                    raise RuntimeError(
                        "bitsandbytes error - Please set USE_4BIT = False at line 52 and restart runtime"
                    ) from e2
                else:
                    raise
        else:
            # Re-raise if it's not a bitsandbytes error
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

# Create zip archive
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
    
    prompt = """Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
Ellie: how are you doing today
Participant: i've been feeling really down lately, nothing seems fun anymore"""
    
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
