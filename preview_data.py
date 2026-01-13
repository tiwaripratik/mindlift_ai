# -*- coding: utf-8 -*-
"""
Helper script to preview and test data processing for PHQ-8 training data.
COLAB-COMPATIBLE VERSION

This script can be run independently to:
1. Test the data processing pipeline
2. Preview generated training pairs
3. Validate data format before training

Usage in Colab:
    # Option 1: Upload your CSV file first, then run:
    !python preview_data.py /content/your_file.csv
    
    # Option 2: Call the function directly (recommended for Colab):
    preview_data_processing('/content/your_file.csv', num_examples=5)
    
    # Option 3: Just run the script - it will auto-detect CSV files in /content/
    exec(open('preview_data.py').read())
"""

import os
import json
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional

# PHQ-8 Questions mapping (same as in finetuning.py)
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
    """Identify which PHQ-8 question the interviewer is asking"""
    question_lower = question_text.lower()
    
    for topic, info in PHQ8_QUESTIONS.items():
        keywords = info["keywords"]
        if any(keyword in question_lower for keyword in keywords):
            return topic
    
    return None


def extract_phq8_answer(response_text: str, topic: str) -> Tuple[Optional[int], float]:
    """Extract PHQ-8 answer score (0-3) from user response with enhanced accuracy"""
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
    """Process conversation data into input/output pairs for training"""
    training_pairs = []
    
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
    
    # Process conversation in pairs (question-answer)
    for i in range(len(conversation_turns) - 1):
        current_turn = conversation_turns[i]
        next_turn = conversation_turns[i + 1]
        
        # Skip if same speaker speaks twice in a row
        if current_turn["speaker"] == next_turn["speaker"]:
            continue
        
        # Identify if current turn is a PHQ-8 question
        question_topic = None
        if current_turn["speaker"].lower() in ["ellie", "interviewer", "assistant"]:
            question_topic = identify_phq8_question(current_turn["text"])
        
        # If it's a PHQ-8 question, extract answer from next turn
        if question_topic and next_turn["speaker"].lower() in ["participant", "user"]:
            answer_score, confidence = extract_phq8_answer(next_turn["text"], question_topic)
            
            if answer_score is not None:
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
    
    # Also create pairs for general depression detection
    # Only create these when there's meaningful PHQ-8 content in conversation
    for i in range(len(conversation_turns)):
        if conversation_turns[i]["speaker"].lower() in ["participant", "user"]:
            # Build context around this response
            context_start = max(0, i - 3)
            context_end = min(len(conversation_turns), i + 1)
            context_turns = conversation_turns[context_start:context_end]
            
            # Check if conversation has PHQ-8 related content
            context_text = " ".join([turn["text"].lower() for turn in context_turns])
            phq8_keywords_in_context = []
            for topic, info in PHQ8_QUESTIONS.items():
                for keyword in info["keywords"]:
                    if keyword in context_text:
                        phq8_keywords_in_context.append(keyword)
            
            # Skip if no PHQ-8 keywords in context (not a PHQ-8 related conversation)
            if not phq8_keywords_in_context:
                continue
            
            context = "\n".join([
                f"{turn['speaker']}: {turn['text']}" 
                for turn in context_turns
            ])
            
            user_response = conversation_turns[i]["text"]
            
            # Check for depression indicators across all PHQ-8 topics
            detected_topics = []
            total_score = 0
            
            for topic, info in PHQ8_QUESTIONS.items():
                answer_score, confidence = extract_phq8_answer(user_response, topic)
                if answer_score is not None and confidence > 0.5:
                    detected_topics.append({
                        "topic": topic,
                        "score": answer_score,
                        "confidence": confidence
                    })
                    total_score += answer_score
            
            # Only create entry if we detected meaningful symptoms (score > 0) or multiple topics
            if detected_topics and (total_score > 0 or len(detected_topics) >= 2):
                # Determine depression severity
                phq8_total = total_score
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
                
                input_text = f"""Analyze the following conversation for depression indicators using PHQ-8 assessment:

Conversation:
{context}

Identify:
1. PHQ-8 symptoms present in the user's responses
2. PHQ-8 scores for each detected symptom
3. Overall depression assessment"""
                
                detected_symptoms = "\n".join([
                    f"- {t['topic']}: Score {t['score']} (confidence: {t['confidence']:.2f})"
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
                    "confidence": sum(t["confidence"] for t in detected_topics) / len(detected_topics)
                })
    
    return training_pairs


def save_training_data(training_pairs: List[Dict], output_path: str):
    """Save training data to JSON file"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(training_pairs)} training pairs to {output_path}")


def preview_data_processing(csv_path: str, num_examples: int = 5):
    """Preview the data processing pipeline"""
    print("=" * 80)
    print("PHQ-8 Training Data Processing Preview")
    print("=" * 80)
    
    # Load data
    print(f"\n1. Loading data from {csv_path}...")
    try:
        df = load_conversation_data(csv_path)
        print(f"   ✓ Loaded {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   First few rows:")
        print(df.head(10).to_string())
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Process data
    print(f"\n2. Processing conversation data...")
    try:
        training_pairs = process_conversation_to_training_pairs(df)
        print(f"   ✓ Created {len(training_pairs)} training pairs")
        
        if len(training_pairs) == 0:
            print("   ⚠ Warning: No training pairs created. Check data format.")
            return
        
        # Show statistics
        topics = {}
        scores = []
        for pair in training_pairs:
            topic = pair.get("topic", "unknown")
            topics[topic] = topics.get(topic, 0) + 1
            if "score" in pair:
                scores.append(pair["score"])
        
        print(f"\n   Statistics:")
        print(f"   - Topics distribution: {topics}")
        if scores:
            print(f"   - Average score: {sum(scores)/len(scores):.2f}")
            print(f"   - Score range: {min(scores)} - {max(scores)}")
        
        # Show examples
        print(f"\n3. Example training pairs (showing first {num_examples}):")
        print("=" * 80)
        for i, pair in enumerate(training_pairs[:num_examples]):
            print(f"\nExample {i+1}:")
            print(f"Topic: {pair.get('topic', 'N/A')}")
            print(f"Score: {pair.get('score', 'N/A')}")
            print(f"Confidence: {pair.get('confidence', 'N/A'):.2f}")
            print(f"\nInput:")
            print(pair['input'][:300] + "..." if len(pair['input']) > 300 else pair['input'])
            print(f"\nOutput:")
            print(pair['output'][:300] + "..." if len(pair['output']) > 300 else pair['output'])
            print("-" * 80)
        
        # Save preview
        output_path = "phq8_training_data_preview.json"
        save_training_data(training_pairs, output_path)
        print(f"\n4. ✓ Saved preview to {output_path}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Colab-friendly: check for uploaded files
    csv_path = None
    
    # Try to find CSV file in common locations
    # First, check sys.argv for CSV paths (but skip kernel files)
    csv_from_args = None
    if len(sys.argv) > 1:
        arg_val = sys.argv[1]
        # Skip kernel/runtime JSON files that Colab/IPython might add
        if not (arg_val.endswith('.json') and ('kernel' in arg_val.lower() or 'runtime' in arg_val.lower())):
            if arg_val.endswith('.csv') or os.path.exists(arg_val):
                csv_from_args = arg_val
    
    possible_paths = [
        csv_from_args,
        "combined_sessions.csv",
        "/content/combined_sessions.csv",
        "/content/drive/MyDrive/combined_sessions.csv",
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        print("=" * 80)
        print("CSV file not found. Please upload your conversation CSV file.")
        print("=" * 80)
        print("\nIn Colab, you can:")
        print("1. Upload file manually: from google.colab import files; files.upload()")
        print("2. Or specify path: !python preview_data.py /content/your_file.csv")
        print("\nTrying to list files in /content...")
        try:
            if os.path.exists("/content"):
                files = [f for f in os.listdir("/content") if f.endswith(".csv")]
                if files:
                    print(f"Found CSV files: {files}")
                    csv_path = f"/content/{files[0]}"
                    print(f"Using: {csv_path}")
        except:
            pass
    
    if csv_path:
        # Safely parse num_examples argument (handle Colab/IPython quirks)
        num_examples = 5  # Default
        if len(sys.argv) > 2:
            try:
                # Try to parse as integer, but skip kernel files
                arg_val = sys.argv[2]
                # Skip kernel/runtime JSON files
                if not (arg_val.endswith('.json') and ('kernel' in arg_val.lower() or 'runtime' in arg_val.lower())):
                    num_examples = int(arg_val)
                    if not (1 <= num_examples <= 100):  # Validate range
                        num_examples = 5
            except (ValueError, IndexError):
                num_examples = 5
        
        preview_data_processing(csv_path, num_examples)
    else:
        print("\nPlease provide a CSV file path or upload one first.")
        print("\nYou can also call the function directly:")
        print("  preview_data_processing('/content/your_file.csv', num_examples=5)")
