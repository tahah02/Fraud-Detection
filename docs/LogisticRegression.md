# Logistic Regression (Meta-Learner) - Documentation

## Overview

Logistic Regression is a linear classification algorithm that predicts probabilities using the logistic (sigmoid) function. In our fraud detection system, Logistic Regression serves as the meta-learner in the stacking architecture, combining outputs from Random Forest and XGBoost to produce the final fraud probability.

## Algorithm Explanation

### How Logistic Regression Works

1. **Linear Combination**: Compute weighted sum of inputs
2. **Sigmoid Transformation**: Convert to probability using sigmoid function
3. **Threshold**: Classify based on probability threshold (typically 0.5)

### Mathematical Foundation

```
z = w0 + w1*x1 + w2*x2 + ... + wn*xn

P(fraud) = 1 / (1 + e^(-z))

Where:
- w0 = bias term
- w1, w2, ..., wn = feature weights
- x1, x2, ..., xn = feature values
- P(fraud) = probability of fraud (0 to 1)
```

### Visual Representation

```
                    Sigmoid Function
Probability
    1.0 │                    ┌────────────
        │                   ╱
        │                  ╱
    0.5 │                 ╱
        │                ╱
        │               ╱
    0.0 │──────────────╱
        └─────────┼─────────────────────────
                  0           Linear Score (z)
```

## Role as Meta-Learner

In our stacking architecture:

```
┌─────────────────────────────────────────┐
│          Base Model Outputs              │
│  ┌─────────────┐  ┌─────────────┐       │
│  │    RF_prob  │  │   XGB_prob  │       │
│  │    (0.75)   │  │   (0.82)    │       │
│  └──────┬──────┘  └──────┬──────┘       │
│         │                │               │
│         └───────┬────────┘               │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │     Logistic Regression         │    │
│  │     Meta-Learner                │    │
│  │                                 │    │
│  │  z = w0 + w1*0.75 + w2*0.82    │    │
│  │  P = sigmoid(z) = 0.79         │    │
│  └─────────────────────────────────┘    │
│                 │                        │
│                 ▼                        │
│         Final Probability: 0.79         │
│         Decision: SUSPICIOUS            │
└─────────────────────────────────────────┘
```

## Configuration in Our System

```python
LogisticRegression(
    random_state=42,         # Reproducibility seed
    max_iter=1000,           # Maximum iterations for convergence
    class_weight='balanced'  # Handle imbalanced fraud data
)
```

### Parameter Explanations

| Parameter | Value | Purpose |
|-----------|-------|---------|
| random_state | 42 | Ensures reproducible results |
| max_iter | 1000 | Sufficient iterations for convergence |
| class_weight | 'balanced' | Adjusts for rare fraud cases |

## Why Logistic Regression for Meta-Learning?

1. **Simplicity**: Simple model prevents overfitting on base model outputs
2. **Interpretability**: Weights show how much each base model contributes
3. **Calibrated Probabilities**: Outputs are well-calibrated probabilities
4. **Fast Training**: Quick to train on 2-dimensional input
5. **Regularization**: Built-in L2 regularization prevents overfitting

## Advantages of Stacking with Logistic Regression

### Over Simple Averaging
- Learns optimal weights instead of equal weighting
- Can capture when one model is better than another
- Accounts for model agreement/disagreement

### Over Complex Meta-Learners
- Lower risk of overfitting
- Faster training and prediction
- More interpretable results

## Training Process

```python
# Get base model predictions
rf_proba = rf_model.predict_proba(X_train)[:, 1]
xgb_proba = xgb_model.predict_proba(X_train)[:, 1]

# Stack predictions as meta-features
meta_features = np.column_stack([rf_proba, xgb_proba])

# Train meta-learner
meta_model = LogisticRegression(class_weight='balanced')
meta_model.fit(meta_features, y_train)
```

## Prediction Process

```python
# Get base model predictions for new transaction
rf_prob = rf_model.predict_proba(features)[:, 1]
xgb_prob = xgb_model.predict_proba(features)[:, 1]

# Combine and predict
meta_features = np.column_stack([rf_prob, xgb_prob])
final_probability = meta_model.predict_proba(meta_features)[:, 1]
```

## Interpreting Weights

After training, the meta-learner has weights like:

```
Intercept (w0): -2.5
RF Weight (w1): 1.8
XGB Weight (w2): 2.2
```

This means:
- XGBoost is weighted slightly higher than Random Forest
- Both contribute positively to fraud probability
- Negative intercept provides base regularization

## Output Interpretation

The final probability is used for decision making:

| Probability Range | Decision | Action |
|-------------------|----------|--------|
| 0.00 - 0.50 | APPROVED | Transaction allowed |
| 0.50 - 0.80 | SUSPICIOUS | User review required |
| 0.80 - 1.00 | REJECTED | Transaction blocked |

## Why This Works Well

The combination of Random Forest + XGBoost + Logistic Regression works well because:

1. **Diverse Base Models**: RF and XGB capture different patterns
2. **Error Correction**: Base models make different types of errors
3. **Optimal Combination**: Meta-learner finds best way to combine
4. **Regularization**: Each layer prevents overfitting
5. **Calibration**: Final probabilities are well-calibrated

## Performance Characteristics

- **Training Time**: Very fast (only 2 features)
- **Prediction Time**: Instant (simple linear computation)
- **Memory Usage**: Minimal (just a few coefficients)
- **Accuracy**: Improves overall ensemble performance
