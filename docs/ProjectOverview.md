# Financial Fraud Detection System - Project Overview

## Introduction

The Financial Fraud Detection System is a machine learning-powered application designed to detect and prevent fraudulent financial transactions in real-time. The system uses a stacked ensemble approach combining Random Forest, XGBoost, and Logistic Regression models to achieve high accuracy in fraud detection.

## Project Objectives

1. **Real-time Fraud Detection**: Analyze transactions as they occur and provide immediate risk assessments
2. **User-specific Risk Profiling**: Calculate personalized transaction limits based on user behavior patterns
3. **Transparent Decision Making**: Provide clear explanations for why transactions are flagged or rejected
4. **User Control**: Allow users to review and approve/reject suspicious transactions

## Key Features

### 1. ML-Powered Fraud Detection
- Stacked ensemble architecture for high accuracy
- 19 engineered features for comprehensive analysis
- Real-time probability scoring for each transaction

### 2. Dynamic Transaction Limits
- Base limit: 30% of account balance
- Leverage: Additional 50% based on risk history
- Limits adjust based on user's fraud history

### 3. Transaction Processing
- **Approved**: Low-risk transactions processed automatically
- **Suspicious**: Medium-risk transactions flagged for user review
- **Rejected**: High-risk transactions blocked automatically

### 4. User Dashboard
- Account summary with current balance
- ML-calculated transaction limits
- Transaction history with risk indicators
- New transaction form

## System Users

The system includes 20 hardcoded user accounts for demonstration:
- ahmed, bilal, sana, fatima, omar, ayesha, hassan, zainab, ali, maryam
- yusuf, khadija, ibrahim, noor, hamza, sara, tariq, amina, khalid, layla

Each user has a unique balance and transaction history.

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.11
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, NumPy

## Model Performance

The stacked model achieves:
- **ROC-AUC Score**: 0.999+
- **Precision**: 98% for fraud detection
- **Recall**: 96% for fraud detection
- **Overall Accuracy**: 99.7%

## Use Cases

1. **Banks and Financial Institutions**: Monitor customer transactions for suspicious activity
2. **E-commerce Platforms**: Protect against fraudulent purchases
3. **Payment Processors**: Real-time transaction screening
4. **Digital Wallets**: Secure peer-to-peer transfers
