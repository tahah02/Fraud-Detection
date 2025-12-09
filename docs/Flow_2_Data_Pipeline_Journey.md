# Flow 2: Data Pipeline Journey (Raw Data to Trained Models)

## Complete Pipeline Flow Diagram

```
START PIPELINE (run_pipeline.py)
         ↓
┌─────────────────────────────────────────────┐
│     STEP 1: GENERATE RAW DATA               │
│     (data_simulator.py)                      │
│                                              │
│     10,000 transactions generated            │
│     5% fraud rate                            │
│     Messy data with noise added              │
│         ↓                                    │
│     OUTPUT: data/raw/transactions_raw.csv   │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│     STEP 2: CLEAN DATA                       │
│     (data_cleaning.py)                       │
│                                              │
│     Remove duplicates                        │
│     Fix missing values                       │
│     Normalize categories                     │
│     Cap outliers                             │
│         ↓                                    │
│     OUTPUT: data/clean/transactions_clean.csv│
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│     STEP 3: FEATURE ENGINEERING              │
│     (features.py)                            │
│                                              │
│     Calculate 19 engineered features         │
│     Historical patterns                      │
│     Behavioral indicators                    │
│     Time-based features                      │
│         ↓                                    │
│     OUTPUT: data/feature_engineered_data.csv │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│     STEP 4: MODEL TRAINING                   │
│     (train.py)                               │
│                                              │
│     Train Random Forest                      │
│     Train XGBoost                            │
│     Train Meta-Learner                       │
│         ↓                                    │
│     OUTPUT: models/*.pkl                     │
└─────────────────────────────────────────────┘
         ↓
      PIPELINE COMPLETE
```

---

## Detailed Step-by-Step Flow

---

## STEP 1: Generate Raw Data

### File: `backend/data_simulator.py`

### Process Flow:
```
START
  ↓
Set random seeds (42) for reproducibility
  ↓
Get user IDs and balances from USER_DATA
  ↓
FOR each of 10,000 transactions:
  │
  ├── Select random user
  │
  ├── Determine if fraud (5% probability)
  │     │
  │     ├── IF FRAUD:
  │     │     • Amount = 40-90% of balance (large)
  │     │     • Merchant = Electronics/Online Shopping/Travel
  │     │     • Channel = Online/ATM
  │     │     • Hour = Night time (10PM-6AM)
  │     │
  │     └── IF LEGITIMATE:
  │           • Amount = Small (exponential distribution)
  │           • Merchant = Any random type
  │           • Channel = Any random channel
  │           • Hour = Any time
  │
  ├── Generate random date within past year
  │
  └── Create transaction record
  ↓
ADD MESSY DATA ARTIFACTS:
  • ~3% missing values
  • ~2% duplicate rows
  • ~15% inconsistent date formats
  • ~1% outliers (extreme amounts)
  • ~2% noise in amount field
  ↓
SHUFFLE all rows randomly
  ↓
SAVE to data/raw/transactions_raw.csv
  ↓
END
```

### Output Sample:
```
transaction_date,user_id,transaction_amount,merchant_type,channel,balance_before,balance_after,is_fraud
2024-03-15 14:30:00,USR001,150.50,Grocery,POS,25000.00,24849.50,0
2024-07-22 02:15:00,USR005,35000.00,Electronics,Online,42000.00,7000.00,1
...
```

---

## STEP 2: Clean Data

### File: `backend/data_cleaning.py`

### Process Flow:
```
LOAD raw data from CSV
  ↓
REMOVE DUPLICATES
  • Count: ~200 duplicates removed
  • Method: drop_duplicates()
  ↓
FIX MISSING VALUES
  │
  ├── Numeric columns (amount, balance):
  │     • Convert to numeric type
  │     • Fill missing with MEDIAN
  │
  ├── Categorical columns (merchant, channel):
  │     • Replace '', 'N/A', 'NULL', 'unknown' with NaN
  │     • Fill merchant_type with 'Unknown'
  │     • Fill channel with 'Online'
  │
  └── Date column:
        • Fill missing with current timestamp
  ↓
NORMALIZE CATEGORIES
  │
  ├── merchant_type:
  │     • Strip whitespace
  │     • Convert to Title Case
  │
  └── channel:
        • Strip whitespace
        • Convert to UPPERCASE
        • Map variations:
            'pos', 'POS', 'Point of Sale' → 'POS'
            'atm', 'ATM', 'Cash' → 'ATM'
            'online', 'Web', 'Internet' → 'ONLINE'
  ↓
CONVERT DATA TYPES
  • transaction_date → datetime
  • transaction_amount → float
  • balance_before → float
  • balance_after → float
  • user_id → string
  • is_fraud → integer (0 or 1)
  ↓
CAP OUTLIERS
  • For amount, balance_before, balance_after:
  • Lower bound = 1st percentile
  • Upper bound = 99th percentile
  • Values outside → clipped to bounds
  ↓
SAVE to data/clean/transactions_clean.csv
  ↓
END
```

### Before vs After:
```
BEFORE (Raw):
  • 10,200 rows (with duplicates)
  • Missing values: ~3%
  • Inconsistent formats
  • Outliers present

AFTER (Clean):
  • ~10,000 rows
  • No missing values
  • Consistent formats
  • Outliers capped
```

---

## STEP 3: Feature Engineering

### File: `backend/features.py`

### Process Flow:
```
LOAD clean data from CSV
  ↓
CALCULATE HISTORICAL FEATURES
  │
  ├── Last_6_Month_Avg:
  │     • Rolling mean over 180 transactions per user
  │     • Window = 180, min_periods = 1
  │
  ├── Current_Month_Cumulative_Sum:
  │     • Group by user + year-month
  │     • Cumulative sum within each month
  │
  ├── Deviation_From_Avg:
  │     • transaction_amount - Last_6_Month_Avg
  │
  ├── Deviation_Ratio:
  │     • transaction_amount / (Last_6_Month_Avg + 1)
  │
  └── Rolling_Std:
        • Rolling standard deviation over 30 transactions
  ↓
CALCULATE BEHAVIORAL FEATURES
  │
  ├── Transaction_Velocity:
  │     • 1 / (hours_since_last_transaction + 1)
  │
  ├── Merchant_Risk_Score:
  │     • Calculate fraud rate per merchant type
  │     • Normalize to 0-1 scale
  │
  ├── Amount_Normalized:
  │     • (amount - user_mean) / (user_std + 1)
  │
  ├── Amount_To_Balance_Ratio:
  │     • amount / (balance_before + 1)
  │
  └── Amount_To_Max_Ratio:
        • amount / (user_max_amount + 1)
  ↓
ADD TIME-BASED FEATURES
  │
  ├── Hour: 0-23
  ├── DayOfWeek: 0-6 (Mon-Sun)
  ├── IsWeekend: 1 if Sat/Sun, else 0
  └── IsNightTime: 1 if 10PM-6AM, else 0
  ↓
ENCODE CATEGORICAL FEATURES
  │
  ├── Channel_Encoded:
  │     • POS → 0
  │     • ATM → 1
  │     • ONLINE → 2
  │
  └── Merchant_Encoded:
        • Each merchant type → unique integer
  ↓
SAVE to data/feature_engineered_data.csv
  ↓
END
```

### Final 19 Features:
```
Original (3):
  1. transaction_amount
  2. balance_before
  3. balance_after

Historical (5):
  4. Last_6_Month_Avg
  5. Current_Month_Cumulative_Sum
  6. Deviation_From_Avg
  7. Deviation_Ratio
  8. Rolling_Std

Behavioral (5):
  9. Transaction_Velocity
  10. Merchant_Risk_Score
  11. Amount_Normalized
  12. Amount_To_Balance_Ratio
  13. Amount_To_Max_Ratio

Time-based (4):
  14. Hour
  15. DayOfWeek
  16. IsWeekend
  17. IsNightTime

Encoded (2):
  18. Channel_Encoded
  19. Merchant_Encoded
```

---

## STEP 4: Model Training

### File: `backend/train.py`

### Process Flow:
```
LOAD feature-engineered data
  ↓
PREPARE TRAINING DATA
  • X = 19 feature columns
  • y = is_fraud column
  • Fill any remaining NaN with 0
  ↓
SPLIT DATA
  • Train: 80% (~8,000 samples)
  • Test: 20% (~2,000 samples)
  • Stratified split (maintains fraud ratio)
  ↓
TRAIN RANDOM FOREST
  │
  ├── Configuration:
  │     • n_estimators = 100
  │     • max_depth = 10
  │     • min_samples_split = 5
  │     • min_samples_leaf = 2
  │     • class_weight = 'balanced'
  │
  └── Output: rf_model (trained)
  ↓
TRAIN XGBOOST
  │
  ├── Calculate scale_pos_weight:
  │     • (count non-fraud) / (count fraud)
  │     • ≈ 19 for 5% fraud rate
  │
  ├── Configuration:
  │     • n_estimators = 100
  │     • max_depth = 6
  │     • learning_rate = 0.1
  │     • scale_pos_weight = ~19
  │
  └── Output: xgb_model (trained)
  ↓
TRAIN META-LEARNER
  │
  ├── Get base model predictions on training data:
  │     • rf_proba = rf_model.predict_proba(X_train)[:, 1]
  │     • xgb_proba = xgb_model.predict_proba(X_train)[:, 1]
  │
  ├── Create meta-features:
  │     • meta_features = [rf_proba, xgb_proba]
  │
  ├── Configuration:
  │     • max_iter = 1000
  │     • class_weight = 'balanced'
  │
  └── Output: meta_model (trained)
  ↓
EVALUATE ON TEST SET
  │
  ├── Get predictions from stacked ensemble
  ├── Calculate metrics:
  │     • Classification report
  │     • Confusion matrix
  │     • ROC-AUC score (~0.92)
  │
  └── Print results
  ↓
SAVE MODELS
  │
  ├── models/base_rf.pkl (~334 KB)
  ├── models/base_xgb.pkl (~124 KB)
  ├── models/stacked_model.pkl (~895 bytes)
  └── models/feature_columns.json
  ↓
PIPELINE COMPLETE
```

---

## Summary: Complete Pipeline

```
run_pipeline.py
      │
      ├── Step 0: Create directories (data/raw, data/clean, models)
      │
      ├── Step 1: generate_raw_transactions()
      │     → data/raw/transactions_raw.csv
      │
      ├── Step 2: clean_data()
      │     → data/clean/transactions_clean.csv
      │
      ├── Step 3: engineer_features()
      │     → data/feature_engineered_data.csv
      │
      └── Step 4: train_full_pipeline()
            → models/base_rf.pkl
            → models/base_xgb.pkl
            → models/stacked_model.pkl
            → models/feature_columns.json
```

### To Run Complete Pipeline:
```bash
python run_pipeline.py
```

### Output Files Created:
```
data/
├── raw/
│   └── transactions_raw.csv      (messy raw data)
├── clean/
│   └── transactions_clean.csv    (cleaned data)
└── feature_engineered_data.csv   (19 features)

models/
├── base_rf.pkl                   (Random Forest)
├── base_xgb.pkl                  (XGBoost)
├── stacked_model.pkl             (Meta-Learner)
└── feature_columns.json          (feature names)
```



### flowchart TD
  A[New Transaction Data] --> B[Feature Extraction: Amount, Deviation_From_Avg, Velocity]
  B --> E[Start XGBoost Training / Apply Trees Sequentially]
  E --> F[Tree 1 (Strong Signals): Focus on large errors (e.g., Amount_To_Balance_Ratio > 0.5)]
  F --> G[Tree 2..N-1 (Medium Corrections): Focus on remaining errors (e.g., Transaction_Velocity, Merchant Risk)]
  G --> H[Tree N (Fine-Tuning): Subtle corrections (e.g., IsNightTime, Rolling Std edgecases)]
  H --> I[Sum Tree Outputs (Additive Model)]
  I --> J[Apply Logistic Transform → Fraud Probability (XGBoost Score)]
  J --> K[Meta-Learner (Logistic Regression) Combines XGBoost Score + Random Forest Score]
  K --> L{Final Decision: Fraud (1) / Safe (0)}
