# Model Performance Analysis & Retraining Guide

## Issues Found from Test Results

Based on running `simulate_conversation.py`, the model has several performance issues:

### 1. **Incorrect Score Assignment**
- **Case 1**: Model gave `interest_pleasure: [0]` when the user said "I've completely lost interest in everything" (should be score 3)
- **Case 4**: Model gave `mood: Score 0` when user said "I've been a bit down lately" (should be score 1-2)

### 2. **Missing Symptoms**
- **Case 3 (Severe Depression)**: Model only detected 3 symptoms (interest, mood, self-worth) but missed:
  - Sleep (score 3)
  - Energy (score 3)
  - Concentration (score 3)
- Total score should be 18/24 (Severe) but model gave 9/24 (Mild)

### 3. **Inconsistent Output Format**
- Model generates different formats for different cases
- Some outputs have structured format, others don't
- Makes it harder to parse and use programmatically

### 4. **Low Detection Accuracy**
- Model missing obvious symptoms
- Not following consistent scoring logic

## Root Causes

1. **Insufficient Training**: Only 3 epochs may not be enough
2. **Training Data Quality**: May not have enough examples or balanced distribution
3. **Output Format Inconsistency**: Training data may have inconsistent formats
4. **Learning Rate**: May be too high, causing unstable learning

## Solutions

### Quick Fix (Try First)
1. **Increase Training Epochs**
   - Already updated: `NUM_TRAIN_EPOCHS = 5` (was 3)
   - Consider: `NUM_TRAIN_EPOCHS = 10` if you have enough data

2. **Lower Learning Rate**
   - Already updated: `LEARNING_RATE = 1e-4` (was 2e-4)
   - This helps with more stable training

3. **Retrain with Current Data**
   - Run: `python finetuning.py`
   - This uses improved parameters

### Better Fix (Recommended)

1. **Analyze Training Data**
   ```bash
   python analyze_training_data.py
   ```
   - Check data distribution
   - Verify score balance
   - Check format consistency

2. **Generate Improved Training Data** (if needed)
   ```bash
   python improve_training_data.py <your_csv_file.csv>
   ```
   - Creates more structured outputs
   - Ensures consistent format
   - Better examples for model to learn

3. **Retrain with Improved Data**
   ```bash
   python finetuning.py
   ```

4. **Test Again**
   ```bash
   python simulate_conversation.py
   ```

### Advanced Fix (If Still Poor)

1. **Increase Training Data**
   - Aim for 1000+ training pairs
   - Ensure balanced distribution (similar counts for scores 0-3)
   - Include all 8 PHQ-8 topics

2. **More Epochs**
   - Set `NUM_TRAIN_EPOCHS = 10`
   - Monitor for overfitting

3. **Adjust LoRA Parameters**
   - Try `LORA_R = 32` (lower rank)
   - Try `LORA_ALPHA = 16` (keep same ratio)

4. **Increase Sequence Length** (if GPU allows)
   - Set `MAX_SEQ_LENGTH = 3072`
   - Helps with longer conversations

## Files Created

1. **`analyze_training_data.py`** - Analyzes training data quality
2. **`improve_training_data.py`** - Generates improved training data
3. **`RETRAINING_GUIDE.md`** - Detailed retraining instructions

## Next Steps

1. ✅ **Already Done**: Updated training parameters in `finetuning.py`
   - Epochs: 3 → 5
   - Learning rate: 2e-4 → 1e-4

2. **You Should Do**:
   - Run `python analyze_training_data.py` to check current data
   - If data looks good, retrain: `python finetuning.py`
   - If data needs improvement, run `python improve_training_data.py` first
   - Test again: `python simulate_conversation.py`

3. **If Still Poor**:
   - Generate more training data
   - Increase epochs to 10
   - Check data extraction logic
   - Consider manual data review

## Expected Improvements

After retraining with improved parameters:
- ✅ More accurate score extraction
- ✅ Better symptom detection
- ✅ Consistent output format
- ✅ Higher overall accuracy

The model should correctly identify:
- Score 3 for "completely lost interest"
- Score 3 for "can't sleep at all"
- Score 3 for "completely drained"
- All symptoms in severe cases

## Testing

After retraining, verify:
- Case 1 (Moderate): Should detect all 6 symptoms with correct scores
- Case 2 (Healthy): Should correctly identify minimal/no depression
- Case 3 (Severe): Should detect all 6 symptoms with score 3 each
- Case 4 (Mild): Should detect mild symptoms with scores 1-2

