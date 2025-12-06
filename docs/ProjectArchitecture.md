# Financial Fraud Detection System - Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                     (Streamlit - app.py)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Login Page  │  │  Dashboard   │  │  Transaction Form    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    model.py                               │  │
│  │  - FraudDetectionModel class                             │  │
│  │  - evaluate_transaction()                                │  │
│  │  - predict_fraud_probability()                           │  │
│  │  - calculate_user_limits()                               │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    utils.py                               │  │
│  │  - User authentication                                    │  │
│  │  - User data management                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODEL LAYER                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  Random Forest │  │    XGBoost     │  │   Logistic     │    │
│  │  (Base Model)  │  │  (Base Model)  │  │  Regression    │    │
│  │                │  │                │  │  (Meta-Learner)│    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘    │
│          │                   │                   │              │
│          └───────────────────┼───────────────────┘              │
│                              ▼                                   │
│                    ┌──────────────────┐                         │
│                    │  Stacked Output  │                         │
│                    │ (Final Prediction)│                        │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Raw Data    │  │ Clean Data   │  │  Feature Engineered  │  │
│  │ (CSV files)  │  │ (CSV files)  │  │     Data (CSV)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
project/
├── app.py                    # Main Streamlit application
├── run_pipeline.py           # Complete pipeline runner
├── backend/
│   ├── __init__.py
│   ├── utils.py              # User management, authentication
│   ├── data_simulator.py     # Raw data generator with noise
│   ├── data_cleaning.py      # Data cleaning pipeline
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training pipeline
│   └── model.py              # Fraud detection model class
├── models/
│   ├── base_rf.pkl           # Random Forest model
│   ├── base_xgb.pkl          # XGBoost model
│   ├── stacked_model.pkl     # Meta-learner (Logistic Regression)
│   └── feature_columns.json  # Feature column names
├── data/
│   ├── raw/
│   │   └── transactions_raw.csv
│   ├── clean/
│   │   └── transactions_clean.csv
│   ├── feature_engineered_data.csv
│   └── transactions_history.csv
└── docs/
    ├── ProjectOverview.md
    ├── ProjectArchitecture.md
    ├── ProjectLogic.md
    ├── RandomForest.md
    ├── XGBoost.md
    └── LogisticRegression.md
```

## Data Flow

### 1. Training Pipeline

```
Raw Data Generator → Data Cleaning → Feature Engineering → Model Training
     ↓                    ↓                  ↓                   ↓
  10,000 txns        Remove noise      19 features         3 models
  with 5% fraud      Normalize         calculated          trained
```

### 2. Transaction Processing Flow

```
User Input → Feature Extraction → Base Models → Meta-Learner → Decision
     ↓              ↓                  ↓             ↓            ↓
  Amount      19 features         RF + XGB        Combine      APPROVE
  Merchant    calculated          probas          scores       SUSPICIOUS
  Channel                                                      REJECT
```

## Component Details

### app.py (Frontend)
- Login page with username/password authentication
- Dashboard with account summary and ML-calculated limits
- Transaction form for new transactions
- Suspicious transaction dialog for user approval
- Transaction history table with status coloring

### backend/model.py (ML Interface)
- `FraudDetectionModel` class loads and manages models
- `prepare_transaction_features()` creates 19 features for new transactions
- `predict_fraud_probability()` runs stacked prediction
- `evaluate_transaction()` returns complete risk assessment
- `calculate_user_limits()` computes personalized transaction limits

### backend/train.py (Training)
- Loads feature-engineered data
- Trains Random Forest base model
- Trains XGBoost base model
- Trains Logistic Regression meta-learner on base predictions
- Saves all models and feature columns

### backend/features.py (Feature Engineering)
- 19 engineered features including:
  - Rolling averages and standard deviations
  - Transaction velocity
  - Merchant risk scores
  - Time-based features
  - User-normalized amounts

## Security Considerations

1. **Password Storage**: Passwords are stored in memory (demo only)
2. **Session Management**: Streamlit session state for user sessions
3. **Input Validation**: Amount and merchant type validation
4. **Audit Trail**: All transactions logged with timestamps
