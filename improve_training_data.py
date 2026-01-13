# -*- coding: utf-8 -*-
"""
Improved Training Data Generator
Creates more structured and consistent training data for better model performance

This script generates training data with:
1. More structured output format
2. Consistent scoring format
3. Better examples for the model to learn from
"""

import os
import json
import pandas as pd
from typing import List, Dict

# Import the extraction functions from finetuning.py
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from finetuning, otherwise define here
try:
    from finetuning import (
        PHQ8_QUESTIONS, 
        extract_phq8_answer,
        identify_phq8_question,
        load_conversation_data
    )
except ImportError:
    print("Warning: Could not import from finetuning.py")
    print("Please ensure finetuning.py is in the same directory")
    sys.exit(1)


def create_structured_output(topic: str, score: int, user_response: str, question: str) -> str:
    """Create a structured, consistent output format"""
    
    phq8_info = PHQ8_QUESTIONS.get(topic, {})
    topic_name = topic.replace('_', ' ').title()
    
    # Map score to frequency description
    score_descriptions = {
        0: "Not at all",
        1: "Several days",
        2: "More than half the days",
        3: "Nearly every day"
    }
    
    frequency_desc = score_descriptions.get(score, "Unknown")
    
    # Structured output format
    output = f"""PHQ-8 Question Identified: {phq8_info.get('question', question)}
Topic: {topic}
User Response: {user_response}
PHQ-8 Answer Score: {score} ({frequency_desc})
Depression Assessment: {'Depression indicators present' if score >= 2 else 'Minimal or no depression indicators'}"""
    
    return output


def create_general_assessment_output(detected_topics: List[Dict], total_score: int, severity: str) -> str:
    """Create structured output for general depression assessment"""
    
    # Build symptom list
    symptom_lines = []
    for topic_data in detected_topics:
        topic = topic_data['topic']
        score = topic_data['score']
        confidence = topic_data.get('confidence', 0.0)
        
        topic_name = topic.replace('_', ' ').title()
        symptom_lines.append(f"- {topic_name}: Score {score} (confidence: {confidence:.2f})")
    
    symptoms_text = "\n".join(symptom_lines)
    
    output = f"""Detected PHQ-8 Symptoms:
{symptoms_text}

Total PHQ-8 Score: {total_score}/24
Depression Assessment: {severity}

Severity Breakdown:
- 0-4: Minimal depression
- 5-9: Mild depression
- 10-14: Moderate depression
- 15-19: Moderately severe depression
- 20-24: Severe depression"""
    
    return output


def improve_training_data(csv_path: str, output_path: str = "./improved_train_data.json"):
    """Generate improved training data with better structure"""
    
    print("=" * 80)
    print("Generating Improved Training Data")
    print("=" * 80)
    
    # Load conversation data
    df = load_conversation_data(csv_path)
    
    training_pairs = []
    conversation_turns = []
    
    # Build conversation turns
    for idx, row in df.iterrows():
        speaker = str(row["speaker"]).strip()
        text = str(row["text"]).strip()
        
        if not text or text.lower() in ["nan", "none", ""]:
            continue
        
        conversation_turns.append({
            "speaker": speaker,
            "text": text,
            "turn": idx
        })
    
    print(f"\nLoaded {len(conversation_turns)} conversation turns")
    
    # Process PHQ-8 question-answer pairs
    phq8_pairs = []
    for i in range(len(conversation_turns) - 1):
        current_turn = conversation_turns[i]
        next_turn = conversation_turns[i + 1]
        
        # Skip if same speaker
        if current_turn["speaker"].lower() == next_turn["speaker"].lower():
            continue
        
        # Check if current turn is interviewer asking PHQ-8 question
        speaker_lower = current_turn["speaker"].lower()
        if any(pattern in speaker_lower for pattern in ["ellie", "interviewer", "assistant", "ai", "bot", "therapist"]):
            question_topic = identify_phq8_question(current_turn["text"])
            
            if question_topic:
                # Check if next turn is participant response
                next_speaker_lower = next_turn["speaker"].lower()
                if any(pattern in next_speaker_lower for pattern in ["participant", "user", "person"]):
                    answer_score, confidence = extract_phq8_answer(next_turn["text"], question_topic)
                    
                    if answer_score is not None:
                        # Build context
                        context_start = max(0, i - 2)
                        context_turns = conversation_turns[context_start:i+2]
                        context = "\n".join([
                            f"{turn['speaker']}: {turn['text']}"
                            for turn in context_turns
                        ])
                        
                        # Create structured input
                        phq8_ref = json.dumps({k: v['question'] for k, v in PHQ8_QUESTIONS.items()}, indent=2)
                        
                        input_text = f"""Analyze the following conversation and identify:
1. Which PHQ-8 question is being asked (if any)
2. Extract the PHQ-8 answer score (0-3) from the user's response
3. Assess depression risk based on PHQ-8 score

Conversation:
{context}

PHQ-8 Questions Reference:
{phq8_ref}"""
                        
                        # Create structured output
                        output_text = create_structured_output(
                            question_topic,
                            answer_score,
                            next_turn["text"],
                            current_turn["text"]
                        )
                        
                        phq8_pairs.append({
                            "input": input_text,
                            "output": output_text,
                            "topic": question_topic,
                            "score": answer_score,
                            "confidence": confidence
                        })
    
    print(f"\nCreated {len(phq8_pairs)} PHQ-8 question-answer pairs")
    
    # Process general depression detection
    general_pairs = []
    for i in range(len(conversation_turns)):
        turn = conversation_turns[i]
        speaker_lower = turn["speaker"].lower()
        
        if any(pattern in speaker_lower for pattern in ["participant", "user", "person"]):
            user_response = turn["text"]
            
            # Build context
            context_start = max(0, i - 3)
            context_end = min(len(conversation_turns), i + 1)
            context_turns = conversation_turns[context_start:context_end]
            context = "\n".join([
                f"{turn['speaker']}: {turn['text']}"
                for turn in context_turns
            ])
            
            # Detect symptoms
            detected_topics = []
            for topic, info in PHQ8_QUESTIONS.items():
                answer_score, confidence = extract_phq8_answer(user_response, topic)
                if answer_score is not None and confidence > 0.4:
                    detected_topics.append({
                        "topic": topic,
                        "score": answer_score,
                        "confidence": confidence
                    })
            
            if detected_topics:
                total_score = sum(t["score"] for t in detected_topics)
                
                # Determine severity
                if total_score <= 4:
                    severity = "Minimal depression"
                elif total_score <= 9:
                    severity = "Mild depression"
                elif total_score <= 14:
                    severity = "Moderate depression"
                elif total_score <= 19:
                    severity = "Moderately severe depression"
                else:
                    severity = "Severe depression"
                
                # Create input
                phq8_ref = json.dumps({k: v['question'] for k, v in PHQ8_QUESTIONS.items()}, indent=2)
                
                input_text = f"""Analyze the following conversation for depression indicators using PHQ-8 assessment:
Conversation: {context}

Identify:
1. PHQ-8 symptoms present in the user's responses
2. PHQ-8 scores for each detected symptom (0-3)
3. Overall depression assessment

PHQ-8 Questions Reference:
{phq8_ref}"""
                
                # Create structured output
                output_text = create_general_assessment_output(detected_topics, total_score, severity)
                
                general_pairs.append({
                    "input": input_text,
                    "output": output_text,
                    "topic": "general",
                    "score": total_score,
                    "confidence": sum(t["confidence"] for t in detected_topics) / len(detected_topics)
                })
    
    print(f"Created {len(general_pairs)} general depression assessment pairs")
    
    # Combine all pairs
    all_pairs = phq8_pairs + general_pairs
    
    print(f"\nTotal training pairs: {len(all_pairs)}")
    
    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved improved training data to: {output_path}")
    
    # Statistics
    from collections import Counter
    topics = Counter(item['topic'] for item in all_pairs)
    scores = Counter(item['score'] for item in all_pairs)
    
    print("\n" + "=" * 80)
    print("Training Data Statistics:")
    print("=" * 80)
    print("\nTopic Distribution:")
    for topic, count in topics.most_common():
        print(f"  {topic}: {count}")
    
    print("\nScore Distribution:")
    for score in sorted(scores.keys()):
        print(f"  Score {score}: {scores[score]}")
    
    print("\n" + "=" * 80)
    
    return all_pairs


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python improve_training_data.py <csv_file_path>")
        print("\nExample:")
        print("  python improve_training_data.py /content/300_conversation.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    improve_training_data(csv_path)

