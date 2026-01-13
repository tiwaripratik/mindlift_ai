# -*- coding: utf-8 -*-
"""
Training Data Analysis and Improvement Script
Analyzes training data quality and suggests improvements

Usage:
    python analyze_training_data.py
"""

import os
import json
import pandas as pd
from collections import Counter
from typing import Dict, List

def analyze_training_data(json_path: str = "./train_data_input_output.json"):
    """Analyze training data quality"""
    
    if not os.path.exists(json_path):
        print(f"❌ Training data file not found: {json_path}")
        print("Please run finetuning.py first to generate training data.")
        return
    
    print("=" * 80)
    print("Training Data Analysis")
    print("=" * 80)
    
    # Load training data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal training pairs: {len(data)}")
    
    # Analyze by topic
    topics = Counter()
    scores = Counter()
    confidences = []
    
    for item in data:
        topic = item.get('topic', 'unknown')
        score = item.get('score')
        confidence = item.get('confidence', 0.0)
        
        topics[topic] += 1
        if score is not None:
            scores[score] += 1
        confidences.append(confidence)
    
    print("\n" + "=" * 80)
    print("Topic Distribution:")
    print("=" * 80)
    for topic, count in topics.most_common():
        print(f"  {topic}: {count}")
    
    print("\n" + "=" * 80)
    print("Score Distribution:")
    print("=" * 80)
    for score in sorted(scores.keys()):
        print(f"  Score {score}: {scores[score]} examples")
    
    print("\n" + "=" * 80)
    print("Confidence Statistics:")
    print("=" * 80)
    if confidences:
        print(f"  Average: {sum(confidences)/len(confidences):.3f}")
        print(f"  Min: {min(confidences):.3f}")
        print(f"  Max: {max(confidences):.3f}")
    
    # Check output format consistency
    print("\n" + "=" * 80)
    print("Output Format Analysis:")
    print("=" * 80)
    
    formats = Counter()
    for item in data:
        output = item.get('output', '')
        # Check for common patterns
        if 'PHQ-8 Score:' in output or 'Score:' in output:
            formats['has_score'] += 1
        if 'Topic:' in output:
            formats['has_topic'] += 1
        if 'Depression Assessment:' in output:
            formats['has_assessment'] += 1
        if 'Confidence:' in output:
            formats['has_confidence'] += 1
    
    print(f"  Contains 'Score': {formats.get('has_score', 0)}")
    print(f"  Contains 'Topic': {formats.get('has_topic', 0)}")
    print(f"  Contains 'Assessment': {formats.get('has_assessment', 0)}")
    print(f"  Contains 'Confidence': {formats.get('has_confidence', 0)}")
    
    # Sample outputs
    print("\n" + "=" * 80)
    print("Sample Outputs:")
    print("=" * 80)
    
    # Show examples for each score
    for score_val in [0, 1, 2, 3]:
        examples = [item for item in data if item.get('score') == score_val]
        if examples:
            print(f"\n--- Score {score_val} Examples ({len(examples)} total) ---")
            for i, ex in enumerate(examples[:2], 1):  # Show first 2
                print(f"\nExample {i}:")
                print(f"Topic: {ex.get('topic')}")
                print(f"Output (first 200 chars): {ex.get('output', '')[:200]}...")
    
    # Check for data quality issues
    print("\n" + "=" * 80)
    print("Data Quality Issues:")
    print("=" * 80)
    
    issues = []
    
    # Check for missing scores
    missing_scores = sum(1 for item in data if item.get('score') is None)
    if missing_scores > 0:
        issues.append(f"  ⚠️  {missing_scores} items missing scores")
    
    # Check for low confidence
    low_confidence = sum(1 for c in confidences if c < 0.5)
    if low_confidence > len(data) * 0.2:  # More than 20% low confidence
        issues.append(f"  ⚠️  {low_confidence} items have low confidence (<0.5)")
    
    # Check score distribution balance
    if scores:
        max_score_count = max(scores.values())
        min_score_count = min(scores.values())
        if max_score_count > min_score_count * 5:  # Unbalanced
            issues.append(f"  ⚠️  Score distribution is unbalanced (max: {max_score_count}, min: {min_score_count})")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  ✓ No major issues detected")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    
    recommendations = []
    
    if scores.get(0, 0) < len(data) * 0.15:  # Less than 15% score 0
        recommendations.append("  • Add more examples with score 0 (healthy responses)")
    
    if scores.get(3, 0) < len(data) * 0.15:  # Less than 15% score 3
        recommendations.append("  • Add more examples with score 3 (severe symptoms)")
    
    if len(data) < 500:
        recommendations.append(f"  • Consider adding more training data (currently {len(data)}, recommend 500+)")
    
    if not recommendations:
        recommendations.append("  • Data looks good! Consider increasing training epochs if model performance is poor")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    analyze_training_data()

