# Flow 1: User Transaction Journey (Login to End)

## Complete User Flow Diagram

```
USER OPENS APP
      ↓
   LOGIN PAGE
      ↓
ENTER CREDENTIALS ──→ INVALID? ──→ ERROR MESSAGE ──→ TRY AGAIN
      ↓
   VALID
      ↓
LOAD DASHBOARD
      ↓
LOAD ML MODELS (RF + XGBoost + Meta-Learner)
      ↓
DISPLAY: Balance, Limits, Transaction Form
      ↓
USER SUBMITS TRANSACTION
      ↓
PREPARE 19 FEATURES
      ↓
┌─────────────────────────────────────────────┐
│         PARALLEL PREDICTIONS                │
│                                             │
│  Random Forest ──→ Probability (e.g. 0.78) │
│  XGBoost ────────→ Probability (e.g. 0.82) │
└─────────────────────────────────────────────┘
      ↓
META-LEARNER COMBINES
      ↓
FINAL PROBABILITY (e.g. 0.83)
      ↓
┌─────────────────────────────────────────────┐
│           DECISION ENGINE                   │
│                                             │
│  ≥ 0.8 ─────────────→ REJECTED             │
│  0.5-0.8 OR > Limit ─→ SUSPICIOUS          │
│  < 0.5 AND ≤ Limit ──→ APPROVED            │
└─────────────────────────────────────────────┘
      ↓
DISPLAY RESULT TO USER
      ↓
LOG TRANSACTION
      ↓
      END
```

---

## Detailed Step-by-Step Flow

### Step 1: User Opens Application
```
Action: User navigates to PayU Fraud Detection System
System: Displays login page (app.py → login_page())
Why: Authentication required before any transaction
```

### Step 2: User Enters Credentials
```
Action: User types username and password (e.g., ahmed/ahmed123)
System: Calls authenticate_user(username, password)
Check: Validates against USERS dictionary in utils.py
```

### Step 3: Authentication Result
```
IF VALID:
  → Set session_state.logged_in = True
  → Set session_state.username = username
  → Redirect to dashboard

IF INVALID:
  → Display error: "Invalid username or password"
  → Stay on login page
```

### Step 4: Dashboard Loads
```
Action: System prepares user dashboard
System Does:
  1. Retrieve user_info from USER_DATA (balance, user_id)
  2. Call fraud_model.load_models() if not loaded
  3. Calculate ML-based transaction limits
  4. Display current balance and limits
  5. Show transaction form
```

### Step 5: ML Models Load
```
Files Loaded:
  • models/base_rf.pkl → Random Forest (100 trees)
  • models/base_xgb.pkl → XGBoost (100 boosted trees)
  • models/stacked_model.pkl → Logistic Regression meta-learner
  • models/feature_columns.json → List of 19 features
```

### Step 6: User Fills Transaction Form
```
User Inputs:
  • Amount: $5,000
  • Merchant Type: Electronics
  • Channel: Online

System Captures:
  • User's current balance
  • User's transaction history
  • Current timestamp
```

### Step 7: Feature Engineering (19 Features Created)
```
From Transaction:
  • transaction_amount = $5,000
  • balance_before = $25,000
  • balance_after = $20,000

From User History:
  • Last_6_Month_Avg = $450
  • Deviation_From_Avg = $4,550
  • Deviation_Ratio = 11.09
  • Rolling_Std = $300
  • Transaction_Velocity = 0.5

Risk Calculations:
  • Merchant_Risk_Score = 0.4 (Electronics is high risk)
  • Amount_Normalized = 15.1 (Z-score)
  • Amount_To_Balance_Ratio = 0.20
  • Amount_To_Max_Ratio = 2.5

Time Features:
  • Hour = 23 (11 PM)
  • DayOfWeek = 5 (Saturday)
  • IsWeekend = 1
  • IsNightTime = 1

Encoded Features:
  • Channel_Encoded = 2 (Online)
  • Merchant_Encoded = 1 (Electronics)
```

### Step 8: Random Forest Prediction
```
Process:
  • 100 trees each analyze 19 features
  • Each tree votes: FRAUD or NOT FRAUD
  • Result: 78 trees vote FRAUD

Output: rf_probability = 0.78
```

### Step 9: XGBoost Prediction
```
Process:
  • 100 boosted trees analyze sequentially
  • Each tree corrects previous errors
  • Accumulated corrections produce final score

Output: xgb_probability = 0.82
```

### Step 10: Meta-Learner Combines
```
Input: [0.78, 0.82]

Calculation:
  z = intercept + w1(0.78) + w2(0.82)
  z = -1.2 + 1.5(0.78) + 2.0(0.82)
  z = 1.61

Output: final_probability = sigmoid(1.61) = 0.833
```

### Step 11: Decision Engine
```
Rules Applied:
  • Probability = 0.833
  • User's limit = $11,250

Check 1: Is 0.833 ≥ 0.8? → YES
  → Status = REJECTED
  → Risk Level = HIGH_RISK
  → Message = "Transaction automatically rejected due to high fraud probability"
```

### Step 12: Display Result
```
User Sees:
  • Red error message: "Transaction Rejected!"
  • Fraud Probability: 83.3%
  • Risk Level: HIGH_RISK
  • Reason: High fraud probability detected
```

### Step 13: Log Transaction
```
Saved to data/transactions_history.csv:
  • timestamp: 2024-12-06 23:45:00
  • user_id: USR001
  • amount: 5000.00
  • merchant: Electronics
  • channel: Online
  • status: REJECTED
  • fraud_prob: 0.833
  • risk_level: HIGH_RISK
```

---

## Alternative Flow: SUSPICIOUS Transaction

```
IF probability = 0.65 (between 0.5 and 0.8)
OR amount > user's limit:

  → Status = SUSPICIOUS
  → Show review dialog to user
  
  User Options:
    [Approve Transaction] → Log as APPROVED_BY_USER
    [Reject Transaction] → Log as REJECTED_BY_USER
    [Cancel] → Return to dashboard
```

---

## Alternative Flow: APPROVED Transaction

```
IF probability = 0.25 (below 0.5)
AND amount ≤ user's limit:

  → Status = APPROVED
  → Show success message
  → Log transaction
  → Update display
```
