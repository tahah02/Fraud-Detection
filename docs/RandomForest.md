# Random Forest Classifier - Documentation

## Overview

Random Forest is an ensemble machine learning algorithm that constructs multiple decision trees during training and outputs the mode (classification) or mean prediction (regression) of the individual trees. In our fraud detection system, Random Forest serves as one of the two base models in the stacking architecture.

## Algorithm Explanation

### How Random Forest Works

1. **Bootstrap Sampling**: Create multiple random subsets of the training data (with replacement)
2. **Decision Tree Construction**: Build a decision tree on each subset
3. **Feature Randomization**: At each node, consider only a random subset of features
4. **Aggregation**: Combine predictions from all trees (majority voting for classification)

### Visual Representation

```
Training Data
     │
     ▼
┌────────────────────────────────────┐
│  Bootstrap Sample 1 → Tree 1      │
│  Bootstrap Sample 2 → Tree 2      │
│  Bootstrap Sample 3 → Tree 3      │
│         ...                        │
│  Bootstrap Sample n → Tree n      │
└────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────┐
│     Aggregation (Majority Vote)    │
│  Tree 1: Fraud                     │
│  Tree 2: Not Fraud                 │
│  Tree 3: Fraud                     │
│  ...                               │
│  Final: FRAUD (if majority)        │
└────────────────────────────────────┘
```

## Configuration in Our System

```python
RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    max_depth=10,            # Maximum depth of each tree
    min_samples_split=5,     # Minimum samples to split a node
    min_samples_leaf=2,      # Minimum samples at leaf nodes
    random_state=42,         # Reproducibility seed
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle imbalanced fraud data
)
```

### Parameter Explanations

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 100 | More trees = more stable predictions |
| max_depth | 10 | Prevents overfitting by limiting tree depth |
| min_samples_split | 5 | Requires 5 samples to create a split |
| min_samples_leaf | 2 | Each leaf must have at least 2 samples |
| class_weight | 'balanced' | Adjusts for rare fraud cases |

## Advantages for Fraud Detection

1. **Handles Imbalanced Data**: With `class_weight='balanced'`, it adjusts for rare fraud cases
2. **Feature Importance**: Can identify which features are most predictive
3. **Non-linear Patterns**: Captures complex relationships in transaction data
4. **Robust to Outliers**: Less sensitive to extreme values than single trees
5. **No Feature Scaling Required**: Works with raw feature values

## Feature Importance (Typical Output)

The Random Forest model typically identifies these as top features:

1. **Deviation_Ratio**: How different is this transaction from average
2. **Amount_To_Balance_Ratio**: Transaction size relative to balance
3. **Transaction_Velocity**: How quickly transactions are occurring
4. **Merchant_Risk_Score**: Historical fraud rate of merchant category
5. **IsNightTime**: Night transactions are higher risk

## How It Contributes to Stacking

In our stacked model:

1. Random Forest outputs probability P_rf for each transaction
2. This probability captures patterns based on:
   - Feature interactions through tree splits
   - Ensemble averaging across 100 trees
3. P_rf is passed to the meta-learner along with XGBoost probability

## Training Process

```python
# Load training data
X_train, y_train = prepare_data(features, labels)

# Train Random Forest
rf_model = RandomForestClassifier(...)
rf_model.fit(X_train, y_train)

# Get probability predictions
rf_proba = rf_model.predict_proba(X_test)[:, 1]
```

## Output Interpretation

The Random Forest outputs a probability between 0 and 1:

- **0.0 - 0.3**: Low fraud probability (likely legitimate)
- **0.3 - 0.6**: Moderate fraud probability (needs review)
- **0.6 - 1.0**: High fraud probability (likely fraudulent)

This probability is combined with XGBoost's probability in the meta-learner for final prediction.

## Performance Characteristics

- **Training Time**: Fast (parallelizable across trees)
- **Prediction Time**: Moderate (must traverse all trees)
- **Memory Usage**: Higher (stores all trees)
- **Accuracy**: High (ensemble reduces variance)
