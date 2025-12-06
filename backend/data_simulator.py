import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

from backend.utils import USER_DATA, MERCHANT_TYPES, CHANNELS

np.random.seed(42)
random.seed(42)


def generate_raw_transactions(num_transactions: int = 10000, 
                               fraud_rate: float = 0.05,
                               output_path: str = "data/raw/transactions_raw.csv") -> pd.DataFrame:
    
    print(f"Generating {num_transactions} raw transactions with {fraud_rate*100}% fraud rate...")
    
    user_ids = [data["user_id"] for data in USER_DATA.values()]
    user_balances = {data["user_id"]: data["balance"] for data in USER_DATA.values()}
    
    start_date = datetime.now() - timedelta(days=365)
    
    transactions = []
    
    for i in range(num_transactions):
        user_id = random.choice(user_ids)
        balance = user_balances[user_id]
        
        is_fraud = 1 if random.random() < fraud_rate else 0
        
        if is_fraud:
            amount = np.random.uniform(balance * 0.4, balance * 0.9)
            merchant_type = random.choice(["Electronics", "Online Shopping", "Travel"])
            channel = random.choice(["Online", "ATM"])
            hour = random.choice([0, 1, 2, 3, 4, 23, 22])
        else:
            amount = np.random.exponential(balance * 0.05)
            amount = min(amount, balance * 0.3)
            merchant_type = random.choice(MERCHANT_TYPES)
            channel = random.choice(CHANNELS)
            hour = random.randint(0, 23)
        
        days_offset = random.randint(0, 365)
        trans_date = start_date + timedelta(days=days_offset, hours=hour, 
                                            minutes=random.randint(0, 59))
        
        balance_before = balance + np.random.uniform(-1000, 5000)
        balance_after = balance_before - amount
        
        transactions.append({
            'transaction_date': trans_date,
            'user_id': user_id,
            'transaction_amount': round(amount, 2),
            'merchant_type': merchant_type,
            'channel': channel,
            'balance_before': round(balance_before, 2),
            'balance_after': round(balance_after, 2),
            'is_fraud': is_fraud
        })
    
    df = pd.DataFrame(transactions)
    
    print("Adding messy data artifacts...")
    df = add_missing_values(df)
    df = add_duplicates(df)
    df = add_inconsistent_formats(df)
    df = add_outliers(df)
    df = add_noise(df)
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nRaw data saved to {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    return df


def add_missing_values(df: pd.DataFrame, missing_rate: float = 0.03) -> pd.DataFrame:
    df = df.copy()
    n_rows = len(df)
    
    for col in ['merchant_type', 'channel', 'balance_before', 'balance_after']:
        n_missing = int(n_rows * missing_rate)
        missing_indices = random.sample(range(n_rows), n_missing)
        
        for idx in missing_indices:
            if random.random() < 0.5:
                df.loc[idx, col] = np.nan
            else:
                df.loc[idx, col] = random.choice(['', 'N/A', 'NULL', 'unknown', None])
    
    print(f"  - Added ~{missing_rate*100}% missing values to categorical columns")
    return df


def add_duplicates(df: pd.DataFrame, duplicate_rate: float = 0.02) -> pd.DataFrame:
    n_duplicates = int(len(df) * duplicate_rate)
    
    duplicate_indices = random.sample(range(len(df)), n_duplicates)
    duplicate_rows = df.iloc[duplicate_indices].copy()
    
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    print(f"  - Added {n_duplicates} duplicate rows")
    return df


def add_inconsistent_formats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    date_formats = [
        lambda d: d.strftime("%Y-%m-%d %H:%M:%S"),
        lambda d: d.strftime("%m/%d/%Y %H:%M"),
        lambda d: d.strftime("%d-%m-%Y"),
        lambda d: d.strftime("%Y/%m/%d"),
        lambda d: str(d)
    ]
    
    for idx in range(len(df)):
        if random.random() < 0.15:
            date_val = df.loc[idx, 'transaction_date']
            if pd.notna(date_val):
                if isinstance(date_val, str):
                    try:
                        date_val = pd.to_datetime(date_val)
                    except:
                        continue
                formatter = random.choice(date_formats)
                df.loc[idx, 'transaction_date'] = formatter(date_val)
    
    channel_variations = {
        'POS': ['POS', 'pos', 'Pos', 'Point of Sale', 'P.O.S'],
        'ATM': ['ATM', 'atm', 'Atm', 'ATM Machine', 'A.T.M'],
        'Online': ['Online', 'online', 'ONLINE', 'Web', 'Internet']
    }
    
    for idx in range(len(df)):
        channel = df.loc[idx, 'channel']
        if pd.notna(channel) and channel in channel_variations:
            if random.random() < 0.1:
                df.loc[idx, 'channel'] = random.choice(channel_variations[channel])
    
    print("  - Added inconsistent date and channel formats")
    return df


def add_outliers(df: pd.DataFrame, outlier_rate: float = 0.01) -> pd.DataFrame:
    df = df.copy()
    n_outliers = int(len(df) * outlier_rate)
    
    outlier_indices = random.sample(range(len(df)), n_outliers)
    
    for idx in outlier_indices:
        if random.random() < 0.5:
            df.loc[idx, 'transaction_amount'] = df.loc[idx, 'transaction_amount'] * random.uniform(10, 100)
        else:
            df.loc[idx, 'transaction_amount'] = random.uniform(-1000, -1)
    
    print(f"  - Added {n_outliers} outlier transactions")
    return df


def add_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for idx in range(len(df)):
        if random.random() < 0.02:
            amount = df.loc[idx, 'transaction_amount']
            if pd.notna(amount):
                df.loc[idx, 'transaction_amount'] = str(amount) + random.choice(['', ' ', 'USD', '$'])
        
        if random.random() < 0.01:
            df.loc[idx, 'user_id'] = df.loc[idx, 'user_id'].lower() if random.random() < 0.5 else df.loc[idx, 'user_id'].upper()
    
    print("  - Added random noise to amount and user_id fields")
    return df


if __name__ == "__main__":
    df = generate_raw_transactions(
        num_transactions=10000,
        fraud_rate=0.05,
        output_path="data/raw/transactions_raw.csv"
    )
    
    print("\nData Summary:")
    print(df.info())
    print("\nSample rows:")
    print(df.head(10))
