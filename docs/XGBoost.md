# XGBoost Classifier - Documentation

## Overview

XGBoost (eXtreme Gradient Boosting) is an advanced gradient boosting algorithm that builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. In our fraud detection system, XGBoost serves as the second base model in the stacking architecture.

## Algorithm Explanation

### How XGBoost Works

1. **Initial Prediction**: Start with a base prediction (usually the average)
2. **Calculate Residuals**: Find the difference between predictions and actual values
3. **Build Tree**: Create a decision tree to predict the residuals
4. **Update Predictions**: Add the new tree's predictions (scaled by learning rate)
5. **Repeat**: Continue adding trees until stopping criteria met

### Visual Representation

```
Initial Prediction (P0)
        │
        ▼
┌──────────────────────────────────────┐
│  Iteration 1:                        │
│  Residuals = Actual - P0             │
│  Tree 1 → Predicts residuals         │
│  P1 = P0 + learning_rate × Tree1     │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Iteration 2:                        │
│  Residuals = Actual - P1             │
│  Tree 2 → Predicts new residuals     │
│  P2 = P1 + learning_rate × Tree2     │
└──────────────────────────────────────┘
        │
        ▼
       ...
        │
        ▼
┌──────────────────────────────────────┐
│  Final Prediction = Sum of all trees │
└──────────────────────────────────────┘
```

## Configuration in Our System

```python
XGBClassifier(
    n_estimators=100,              # Number of boosting rounds
    max_depth=6,                   # Maximum depth of each tree
    learning_rate=0.1,             # Step size shrinkage
    scale_pos_weight=19,           # Weight for positive class (fraud)
    random_state=42,               # Reproducibility seed
    n_jobs=-1,                     # Use all CPU cores
    use_label_encoder=False,       # Disable deprecated encoder
    eval_metric='logloss'          # Evaluation metric
)
```

### Parameter Explanations

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 100 | Number of trees to build sequentially |
| max_depth | 6 | Shallower than Random Forest (prevents overfitting) |
| learning_rate | 0.1 | Controls contribution of each tree |
| scale_pos_weight | ~19 | Ratio of negative to positive cases |
| eval_metric | 'logloss' | Logarithmic loss for binary classification |

### Scale Positive Weight Calculation

```python
scale_pos_weight = (num_non_fraud) / (num_fraud)
# Example: 9581 / 508 ≈ 19
```

This tells XGBoost that fraudulent transactions are 19x more important to classify correctly.

## Key Features of XGBoost

1. **Regularization**: L1 and L2 regularization prevent overfitting
2. **Handling Missing Values**: Built-in support for missing data
3. **Parallel Processing**: Efficient multi-threaded implementation
4. **Tree Pruning**: Uses max_depth to control tree complexity
5. **Cross-Validation**: Built-in support for model validation

## Advantages for Fraud Detection

1. **Sequential Learning**: Each tree focuses on hard-to-classify transactions
2. **Class Imbalance Handling**: scale_pos_weight addresses rare fraud cases
3. **Feature Interactions**: Captures complex patterns in transaction data
4. **Gradient-Based**: Optimizes directly for classification accuracy
5. **Speed**: Highly optimized C++ implementation

## How It Differs from Random Forest

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| Training | Trees trained independently (parallel) | Trees trained sequentially |
| Focus | All data points equally | Focus on misclassified points |
| Depth | Deeper trees (10) | Shallower trees (6) |
| Regularization | Implicit (via ensemble) | Explicit (L1, L2) |
| Variance | Reduces variance | Reduces bias |

## How It Contributes to Stacking

In our stacked model:

1. XGBoost outputs probability P_xgb for each transaction
2. This probability captures:
   - Gradient-boosted patterns from sequential learning
   - Focus on difficult-to-classify transactions
3. P_xgb is passed to the meta-learner along with Random Forest probability

## Training Process

```python
# Calculate class weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
xgb_model.fit(X_train, y_train)

# Get probability predictions
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
```

## Output Interpretation

The XGBoost model outputs a probability between 0 and 1:

- **0.0 - 0.3**: Low fraud probability (likely legitimate)
- **0.3 - 0.6**: Moderate fraud probability (needs review)
- **0.6 - 1.0**: High fraud probability (likely fraudulent)

## Complementary to Random Forest

XGBoost and Random Forest complement each other:

1. **Random Forest**: Better at capturing independent patterns, reduces variance
2. **XGBoost**: Better at capturing sequential patterns, reduces bias

By combining both in a stacking ensemble, we get the benefits of both approaches.

## Performance Characteristics

- **Training Time**: Moderate (sequential tree building)
- **Prediction Time**: Fast (efficient tree traversal)
- **Memory Usage**: Lower than Random Forest
- **Accuracy**: Very high (state-of-the-art for tabular data)
