# Version 3 - Linear Regression with Train/Test Split

⚠️ **This version adds train/test split but has a critical regression bug! **

## Key Addition

**Train/Test Split** for proper model evaluation:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## The Problem

Despite adding train/test split, this version **forgot to normalize the target variable (y)**, causing it to regress back to the problems of Version 2.2! 

```python
# What it does (WRONG):
y = data['Sales'].values.reshape(-1, 1)
# No normalization!

# What it should do (from Version 2.3):
y = data['Sales'].values.reshape(-1, 1)
y = (y - y.mean()) / y.std()  # ← Missing this!
```

## Changes from Version 2.3

| Aspect | Version 2.3 | Version 3 |
|--------|-------------|-----------|
| Train/Test Split | ❌ No | ✅ Yes |
| Target Normalization | ✅ Yes | ❌ **No** (regression!) |
| Evaluation | On all data | Separate train/test |
| R² Score | 0.897 | -5.31 (test), -6.77 (train) |

## Files

- `data_preprocessing.py` - **Train/test split** but missing y normalization
- `linear_regression.py` - Same model as Version 2.3
- `metrics.py` - **Train/test comparison functions**
- `train. py` - Training with separate train/test evaluation
- `visualization.py` - **Train vs test plots**
- `train_with_visualization.py` - Complete pipeline
- `README.md` - This file

## Usage

```python
from data_preprocessing import load_and_split_data
from linear_regression import LinearRegression
from metrics import evaluate_model, compare_train_test_performance

# Load and split data
X_train, X_test, y_train, y_test, X_mean, X_std = load_and_split_data(
    'Advertising.csv'
)

# Train model
model = LinearRegression(learn_rate=0.02, iter=80000)
model.fit(X_train, y_train)

# Evaluate on both sets
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_metrics = evaluate_model(y_train, train_pred, "Train")
test_metrics = evaluate_model(y_test, test_pred, "Test")

compare_train_test_performance(train_metrics, test_metrics)
```

## Results

```
Loss at iteration 0: 112.37312500000002
Loss at iteration 5000: 100.75756471154071
...  (loss plateaus - not converging properly)

Training Set:
Mean Squared Error: [high value]
R² Score: -6.769632441971426  ← ❌ Negative! 

Test Set:
Mean Squared Error: 199.23399954062228
R² Score: -5.312145604162146  ← ❌ Negative!
```

## What Went Wrong

1. ❌ **Forgot target normalization** from Version 2.3
2. ❌ **Numerical instability** returned
3. ❌ **Negative R² scores** on both train and test
4. ✅ Train/test split is good (but not enough!)

## The Fix

Version 3.1 should combine: 
- ✅ Train/test split (from Version 3)
- ✅ Target normalization (from Version 2.3)

```python
# In load_and_split_data():
y_mean = y_train.mean()
y_std = y_train. std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std  # Use train stats!
```

## Key Learning

🎓 **Important Lesson**: When adding new features (like train/test split), don't forget previous improvements!  All enhancements must be carried forward: 

1. Float64 precision ✓
2. Corrected loss function ✓
3. Corrected gradient ✓
4. **Target normalization** ← Forgot this! 
5. Train/test split ✓

## Comparison

| Version | Train/Test | Y Normalized | R² Score |
|---------|------------|--------------|----------|
| 2.2 | ❌ | ❌ | -0.73 |
| 2.3 | ❌ | ✅ | **0.897** |
| 3.0 | ✅ | ❌ | -5.31/-6.77 |
| 3.1 | ✅ | ✅ | Should be ~0.89 |

## Next Steps

See **Version 3.1** for the corrected version that combines: 
- Train/test split
- Target normalization
- Should achieve R² ~ 0.89 on test set

---

**⚠️ This version demonstrates why systematic testing and code review are important!**
