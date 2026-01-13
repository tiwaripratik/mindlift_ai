# Critical Issues Found - Model Performance Analysis

## Problem Summary

After retraining, the model is **WORSE** than before:
- ❌ Everything gets score 0
- ❌ Only detects "mood" topic
- ❌ Missing all other symptoms (interest, sleep, energy, appetite, self-worth, concentration)
- ❌ Even severe depression cases get "Minimal depression" assessment

## Root Causes

### 1. **Training Data Imbalance (Most Likely)**
The training data probably has too many score 0 examples, causing the model to learn:
- "Always predict score 0"
- "Only detect mood topic"
- "Be conservative"

**Evidence**: Model gives score 0 for everything, even severe cases.

### 2. **Prompt Format Mismatch (Fixed)**
The test prompt format didn't exactly match training format.
- ✅ **FIXED**: Updated `simulate_conversation.py` to match training format

### 3. **Model Not Learning Extraction Logic**
The model is generating text but not following the structured format it was trained on.

### 4. **Insufficient Training Data**
If training data is too small or unbalanced, model won't learn properly.

## Diagnostic Steps

### Step 1: Check Training Data
```bash
python diagnose_training_data.py
```

This will show:
- Score distribution (how many 0, 1, 2, 3)
- Topic distribution
- Output format consistency
- Potential issues

### Step 2: Check if Too Many Score 0 Examples

If you see:
- Score 0: 60%+ of examples
- Score 3: <10% of examples

Then the model is learning to predict 0.

## Solutions

### Solution 1: Fix Training Data (CRITICAL)

**Option A: Filter Score 0 Examples**
```python
# In finetuning.py, modify process_conversation_to_training_pairs()
# Only include score 0 if it's a clear healthy response
# Remove ambiguous score 0 examples
```

**Option B: Add More Non-Zero Examples**
- Ensure at least 30% of examples have scores 1-3
- Ensure balanced distribution across all scores

**Option C: Check extract_phq8_answer() Logic**
The function might be too conservative and returning 0 too often.

### Solution 2: Adjust Training Data Generation

Modify `finetuning.py` to:
1. Be less lenient with score 0 detection
2. Ensure score 0 examples are clearly positive/healthy responses
3. Include more examples with scores 1-3

### Solution 3: Improve Training Parameters

Try:
```python
NUM_TRAIN_EPOCHS = 10  # More epochs
LEARNING_RATE = 5e-5   # Lower learning rate
```

### Solution 4: Use Few-Shot Examples in Prompt

Add examples to the prompt to guide the model:
```python
prompt = f"""Analyze the following conversation...

Example 1:
User: "I've completely lost interest in everything"
Score: 3 (Nearly every day)

Example 2:
User: "I feel down sometimes"
Score: 1 (Several days)

Now analyze this conversation:
{conversation_text}
"""
```

## Immediate Actions

1. ✅ **DONE**: Fixed prompt format mismatch in `simulate_conversation.py`

2. **RUN THIS NOW**:
   ```bash
   python diagnose_training_data.py
   ```
   This will show you exactly what's wrong with your training data.

3. **Based on diagnosis**:
   - If too many score 0 → Filter or rebalance data
   - If too little data → Add more examples
   - If format issues → Fix output format consistency

4. **Retrain** with fixed data:
   ```bash
   python finetuning.py
   ```

5. **Test again**:
   ```bash
   python simulate_conversation.py
   ```

## Expected Fix

After fixing training data:
- Model should detect multiple symptoms
- Scores should be accurate (3 for severe, 1-2 for mild)
- Should detect all 8 PHQ-8 topics, not just "mood"

## Why This Happened

The low training loss (0.2-0.3) was misleading. The model learned to:
- Predict the most common class (score 0)
- Use simple patterns (only mood detection)
- Avoid making mistakes by being conservative

Low loss ≠ Good performance if the model learns the wrong thing!

