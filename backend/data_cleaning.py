import pandas as pd
import numpy as np
from datetime import datetime
import os


def load_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)
    df = df.drop_duplicates()
    removed = initial_count - len(df)
    print(f"Removed {removed} duplicate rows")
    return df


def fix_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df['transaction_amount'] = df['transaction_amount'].fillna(df['transaction_amount'].median())
    df['balance_before'] = df['balance_before'].fillna(df['balance_before'].median())
    df['balance_after'] = df['balance_after'].fillna(df['balance_after'].median())
    df['merchant_type'] = df['merchant_type'].fillna('Unknown')
    df['channel'] = df['channel'].fillna('Online')
    df['user_id'] = df['user_id'].fillna('UNKNOWN')
    df['is_fraud'] = df['is_fraud'].fillna(0)
    
    if df['transaction_date'].isna().any():
        df['transaction_date'] = df['transaction_date'].fillna(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return df


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df['merchant_type'] = df['merchant_type'].str.strip().str.title()
    df['channel'] = df['channel'].str.strip().str.upper()
    
    channel_mapping = {
        'POS': 'POS', 'POINT OF SALE': 'POS', 'TERMINAL': 'POS',
        'ATM': 'ATM', 'CASH': 'ATM', 'WITHDRAWAL': 'ATM',
        'ONLINE': 'ONLINE', 'WEB': 'ONLINE', 'INTERNET': 'ONLINE', 'ECOMMERCE': 'ONLINE'
    }
    df['channel'] = df['channel'].map(lambda x: channel_mapping.get(x, 'ONLINE'))
    
    return df


def cap_outliers(df: pd.DataFrame, column: str, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['transaction_date'] = df['transaction_date'].fillna(datetime.now())
    
    df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0).astype(float)
    df['balance_before'] = pd.to_numeric(df['balance_before'], errors='coerce').fillna(0).astype(float)
    df['balance_after'] = pd.to_numeric(df['balance_after'], errors='coerce').fillna(0).astype(float)
    
    df['user_id'] = df['user_id'].astype(str)
    
    df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)
    
    return df


def clean_data(raw_filepath: str, clean_filepath: str) -> pd.DataFrame:
    print("Loading raw data...")
    df = load_raw_data(raw_filepath)
    print(f"Loaded {len(df)} rows")
    
    print("\nRemoving duplicates...")
    df = remove_duplicates(df)
    
    print("\nFixing missing values...")
    df = fix_missing_values(df)
    
    print("\nNormalizing categories...")
    df = normalize_categories(df)
    
    print("\nConverting data types...")
    df = convert_data_types(df)
    
    print("\nCapping outliers...")
    df = cap_outliers(df, 'transaction_amount')
    df = cap_outliers(df, 'balance_before')
    df = cap_outliers(df, 'balance_after')
    
    os.makedirs(os.path.dirname(clean_filepath), exist_ok=True)
    df.to_csv(clean_filepath, index=False)
    print(f"\nCleaned data saved to {clean_filepath}")
    print(f"Final dataset: {len(df)} rows")
    
    return df


if __name__ == "__main__":
    clean_data("data/raw/transactions_raw.csv", "data/clean/transactions_clean.csv")
