# Financial Fraud Detection System - Project Logic

## Core Logic Flow

### 1. Transaction Evaluation Logic

When a user submits a transaction, the system follows this logic:

```
1. Receive transaction details (amount, merchant, channel)
2. Calculate 19 features for the transaction
3. Run features through Random Forest → get probability P1
4. Run features through XGBoost → get probability P2
5. Combine P1 and P2 → run through Logistic Regression → get final probability P
6. Apply decision rules:
   - P >= 0.8: REJECT (automatic block)
   - P >= 0.5 OR amount > limit: SUSPICIOUS (user review)
   - P < 0.5 AND amount <= limit: APPROVE (automatic approval)
```

### 2. Feature Engineering Logic

The system calculates 19 features from each transaction:

| Feature | Description | Formula |
|---------|-------------|---------|
| transaction_amount | Raw amount | Direct value |
| balance_before | Balance before transaction | Direct value |
| balance_after | Balance after transaction | balance_before - amount |
| Last_6_Month_Avg | Average transaction amount (6 months) | Rolling mean (180 days) |
| Current_Month_Cumulative_Sum | Total spending this month | Cumulative sum |
| Deviation_From_Avg | Difference from average | amount - Last_6_Month_Avg |
| Deviation_Ratio | Ratio to average | amount / (Last_6_Month_Avg + 1) |
| Rolling_Std | Transaction amount volatility | Rolling std (30 days) |
| Transaction_Velocity | Speed of transactions | 1 / (hours_since_last + 1) |
| Merchant_Risk_Score | Merchant category risk | Historical fraud rate |
| Amount_Normalized | Z-score of amount | (amount - mean) / std |
| Amount_To_Balance_Ratio | Proportion of balance | amount / balance |
| Amount_To_Max_Ratio | Compared to max transaction | amount / max_amount |
| Hour | Hour of transaction | 0-23 |
| DayOfWeek | Day of week | 0-6 (Mon-Sun) |
| IsWeekend | Weekend flag | 1 if Sat/Sun, else 0 |
| IsNightTime | Night transaction | 1 if 22:00-06:00, else 0 |
| Channel_Encoded | Channel numeric | POS=0, ATM=1, Online=2 |
| Merchant_Encoded | Merchant numeric | 0-10 based on category |

### 3. Limit Calculation Logic

User transaction limits are calculated dynamically:

```python
base_limit = balance * 0.30          # 30% of current balance
risk_factor = 1 - fraud_history      # Lower if user has fraud history
leverage = base_limit * 0.50 * risk_factor  # Up to 50% additional
total_limit = base_limit + leverage  # Final limit
```

Example:
- User balance: $25,000
- No fraud history (fraud_history = 0)
- base_limit = $25,000 * 0.30 = $7,500
- risk_factor = 1 - 0 = 1.0
- leverage = $7,500 * 0.50 * 1.0 = $3,750
- total_limit = $7,500 + $3,750 = $11,250

### 4. Fraud Detection Signals

The model looks for these fraud indicators:

**High Risk Indicators:**
- Large transaction amounts relative to user history
- Transactions at unusual hours (10PM - 6AM)
- High-risk merchants (Electronics, Online Shopping)
- Rapid succession of transactions
- Amounts exceeding balance ratio thresholds

**Low Risk Indicators:**
- Transaction amounts within normal range
- Regular business hours
- Low-risk merchants (Grocery, Utilities)
- Consistent transaction patterns
- Within historical spending patterns

### 5. Stacking Model Logic

The stacking approach works as follows:

1. **Level 0 (Base Models):**
   - Random Forest: Captures non-linear patterns, handles imbalanced data
   - XGBoost: Gradient boosting for complex feature interactions

2. **Level 1 (Meta-Learner):**
   - Logistic Regression: Combines base model outputs
   - Input: [RF_probability, XGB_probability]
   - Output: Final fraud probability

3. **Ensemble Benefits:**
   - Reduces overfitting from any single model
   - Combines strengths of different algorithms
   - More robust predictions

### 6. Data Cleaning Logic

Raw data is cleaned through these steps:

```
1. Remove duplicate transactions
2. Convert data types (strings → numbers, dates)
3. Handle missing values:
   - Numeric: Replace with median
   - Categorical: Replace with mode or 'Unknown'
4. Normalize categories:
   - Channel: POS, ATM, ONLINE (standardized)
   - Merchant: Title case
5. Cap outliers:
   - Lower bound: 1st percentile
   - Upper bound: 99th percentile
```

### 7. Session Management Logic

```
User opens app → Check session state
   ↓
If not logged in → Show login page
   ↓
User submits credentials → Validate against USERS dict
   ↓
If valid → Set session_state.logged_in = True
         → Store username in session_state
         → Redirect to dashboard
   ↓
If invalid → Show error message
```

### 8. Transaction History Logic

All transactions are logged to `data/transactions_history.csv`:

```python
log_entry = {
    'timestamp': current_time,
    'user_id': user_id,
    'amount': transaction_amount,
    'merchant': merchant_type,
    'channel': channel,
    'status': final_status,  # APPROVED, REJECTED, APPROVED_BY_USER, REJECTED_BY_USER
    'fraud_prob': probability,
    'risk_level': risk_level  # LOW_RISK, MEDIUM_RISK, HIGH_RISK
}
```

This history is used to:
1. Display user's transaction history
2. Calculate user-specific statistics
3. Improve future predictions
