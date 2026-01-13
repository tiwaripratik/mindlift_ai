# -*- coding: utf-8 -*-
"""
RETRAINING GUIDE - Step by Step Instructions

Based on test results showing poor model performance, follow these steps to retrain:

ISSUES FOUND:
1. Model giving incorrect scores (e.g., score 0 for "completely lost interest")
2. Model missing symptoms (only detecting 3 out of 6)
3. Inconsistent output format
4. Low accuracy in score extraction

SOLUTIONS:

STEP 1: Analyze Current Training Data
---------------------------------------
Run: python analyze_training_data.py

This will show:
- Number of training examples
- Score distribution (should be balanced)
- Topic distribution
- Format consistency issues

STEP 2: Generate Improved Training Data (Optional but Recommended)
-------------------------------------------------------------------
Run: python improve_training_data.py <your_csv_file.csv>

This creates training data with:
- More structured output format
- Consistent formatting
- Better examples

STEP 3: Update Training Parameters
-----------------------------------
Edit finetuning.py:
- NUM_TRAIN_EPOCHS: 3 → 5-10 (more epochs for better learning)
- LEARNING_RATE: 2e-4 → 1e-4 (lower LR for stability)
- MAX_SEQ_LENGTH: 2048 → 3072 (if GPU memory allows)

STEP 4: Ensure Enough Training Data
-------------------------------------
Aim for:
- Minimum 500 training pairs
- Balanced score distribution (similar counts for 0, 1, 2, 3)
- All 8 PHQ-8 topics well represented
- Both question-answer pairs AND general assessment pairs

STEP 5: Retrain Model
----------------------
Run: python finetuning.py

Monitor:
- Training loss should decrease consistently
- Validation loss should not increase (overfitting)
- Training should complete all epochs

STEP 6: Test Model
-------------------
Run: python simulate_conversation.py

Check:
- Scores are correct
- All symptoms detected
- Output format is consistent
- Assessment matches severity

STEP 7: If Still Poor Performance
-----------------------------------
Consider:
1. Add more training data (1000+ examples)
2. Increase epochs to 10
3. Check if data extraction logic is correct
4. Manually review training examples
5. Consider fine-tuning base model first
6. Try different LoRA parameters (r=32, alpha=16)

QUICK FIX OPTIONS:
------------------
Option 1: Increase epochs only
- Set NUM_TRAIN_EPOCHS = 10
- Retrain with same data

Option 2: Improve data quality
- Run improve_training_data.py
- Use improved data for training

Option 3: Both (Recommended)
- Improve data + increase epochs
- Best chance of success
"""

import json

GUIDE = {
    "step_1": {
        "title": "Analyze Current Training Data",
        "command": "python analyze_training_data.py",
        "purpose": "Understand data quality and distribution"
    },
    "step_2": {
        "title": "Generate Improved Training Data",
        "command": "python improve_training_data.py <csv_file>",
        "purpose": "Create better structured training data"
    },
    "step_3": {
        "title": "Update Training Parameters",
        "changes": {
            "NUM_TRAIN_EPOCHS": "3 → 5-10",
            "LEARNING_RATE": "2e-4 → 1e-4",
            "MAX_SEQ_LENGTH": "2048 → 3072 (if possible)"
        }
    },
    "step_4": {
        "title": "Retrain Model",
        "command": "python finetuning.py"
    },
    "step_5": {
        "title": "Test Model",
        "command": "python simulate_conversation.py"
    }
}

print("=" * 80)
print("RETRAINING GUIDE")
print("=" * 80)
print(json.dumps(GUIDE, indent=2))
