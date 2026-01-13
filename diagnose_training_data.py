# -*- coding: utf-8 -*-
"""
Diagnostic Script - Find Why Model Performs Poorly

This script checks:
1. Training data distribution
2. Prompt format consistency
3. Score distribution in training data
"""

import json
import os
from collections import Counter

def diagnose_training_data(json_path: str = "./phq8_training_data.json"):
    """Diagnose training data issues"""
    
    print("=" * 80)
    print("TRAINING DATA DIAGNOSIS")
    print("=" * 80)
    
    if not os.path.exists(json_path):
        print(f"\n❌ Training data file not found: {json_path}")
        print("Please run finetuning.py first to generate training data.")
        return
    
    # Load training data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal training pairs: {len(data)}")
    
    # Analyze score distribution
    scores = Counter()
    topics = Counter()
    output_formats = Counter()
    
    for item in data:
        score = item.get('score')
        topic = item.get('topic', 'unknown')
        output = item.get('output', '')
        
        if score is not None:
            scores[score] += 1
        topics[topic] += 1
        
        # Check output format
        if 'Score' in output:
            output_formats['has_score'] += 1
        if 'Topic:' in output:
            output_formats['has_topic'] += 1
        if 'Depression Assessment:' in output or 'Depression Assessment' in output:
            output_formats['has_assessment'] += 1
    
    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION:")
    print("=" * 80)
    for score in sorted(scores.keys()):
        count = scores[score]
        percentage = (count / len(data)) * 100
        print(f"  Score {score}: {count} ({percentage:.1f}%)")
    
    # Check for imbalance
    if scores.get(0, 0) > len(data) * 0.5:
        print(f"\n⚠️  WARNING: {scores.get(0, 0)} out of {len(data)} examples are score 0 ({scores.get(0, 0)/len(data)*100:.1f}%)")
        print("   This is causing the model to learn to predict score 0 for everything!")
    
    print("\n" + "=" * 80)
    print("TOPIC DISTRIBUTION:")
    print("=" * 80)
    for topic, count in topics.most_common():
        print(f"  {topic}: {count}")
    
    print("\n" + "=" * 80)
    print("OUTPUT FORMAT CHECK:")
    print("=" * 80)
    print(f"  Contains 'Score': {output_formats.get('has_score', 0)}")
    print(f"  Contains 'Topic:': {output_formats.get('has_topic', 0)}")
    print(f"  Contains 'Assessment': {output_formats.get('has_assessment', 0)}")
    
    # Sample outputs
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS:")
    print("=" * 80)
    
    # Show examples for each score
    for score_val in [0, 1, 2, 3]:
        examples = [item for item in data if item.get('score') == score_val]
        if examples:
            print(f"\n--- Score {score_val} Example ---")
            ex = examples[0]
            print(f"Input (first 200 chars): {ex.get('input', '')[:200]}...")
            print(f"\nOutput:")
            print(ex.get('output', '')[:500])
            print("-" * 80)
    
    # Check for issues
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES:")
    print("=" * 80)
    
    issues = []
    
    if scores.get(0, 0) > len(data) * 0.6:
        issues.append(f"❌ Too many score 0 examples ({scores.get(0, 0)}/{len(data)}) - model learns to predict 0")
    
    if scores.get(3, 0) < len(data) * 0.1:
        issues.append(f"❌ Too few score 3 examples ({scores.get(3, 0)}/{len(data)}) - model doesn't learn severe cases")
    
    if len(data) < 200:
        issues.append(f"❌ Too little training data ({len(data)}) - recommend at least 500+ examples")
    
    # Check if only "general" topic
    if topics.get('general', 0) > len(data) * 0.8:
        issues.append(f"❌ Too many 'general' topic examples ({topics.get('general', 0)}/{len(data)}) - not enough specific PHQ-8 examples")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No major issues detected")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    recommendations = []
    
    if scores.get(0, 0) > len(data) * 0.5:
        recommendations.append("• Filter out some score 0 examples OR add more non-zero examples")
        recommendations.append("• Check if extract_phq8_answer() is too conservative")
    
    if len(data) < 500:
        recommendations.append(f"• Add more training data (currently {len(data)}, aim for 1000+)")
    
    if topics.get('general', 0) > len(data) * 0.7:
        recommendations.append("• Ensure more specific PHQ-8 question-answer pairs in training data")
    
    if not recommendations:
        recommendations.append("• Data looks balanced - issue might be in model training or prompt format")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    diagnose_training_data()

