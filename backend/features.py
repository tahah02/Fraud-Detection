import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def calculate_last_6_month_avg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['user_id', 'transaction_date'])
    
    def rolling_avg(group):
        group['Last_6_Month_Avg'] = group['transaction_amount'].rolling(
            window=180, min_periods=1
        ).mean()
        return group
    
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    df = df.groupby('user_id', group_keys=False).apply(rolling_avg, include_groups=False)
    df['Last_6_Month_Avg'] = df['Last_6_Month_Avg'].fillna(df['transaction_amount'])
    
    return df


def calculate_current_month_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['year_month'] = df['transaction_date'].dt.to_period('M')
    
    df = df.sort_values(['user_id', 'year_month', 'transaction_date'])
    
    df['Current_Month_Cumulative_Sum'] = df.groupby(['user_id', 'year_month'])['transaction_amount'].cumsum()
    
    df = df.drop('year_month', axis=1)
    
    return df


def calculate_deviation_from_avg(df: pd.DataFrame) -> pd.DataFrame:
    df['Deviation_From_Avg'] = df['transaction_amount'] - df['Last_6_Month_Avg']
    df['Deviation_Ratio'] = df['transaction_amount'] / (df['Last_6_Month_Avg'] + 1)
    
    return df


def calculate_rolling_std(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.sort_values(['user_id', 'transaction_date'])
    
    def rolling_std(group):
        group['Rolling_Std'] = group['transaction_amount'].rolling(
            window=window, min_periods=1
        ).std()
        return group
    
    df = df.groupby('user_id', group_keys=False).apply(rolling_std, include_groups=False)
    df['Rolling_Std'] = df['Rolling_Std'].fillna(0)
    
    return df


def calculate_transaction_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['user_id', 'transaction_date'])
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    def calc_velocity(group):
        group['time_diff'] = group['transaction_date'].diff().dt.total_seconds() / 3600
        group['Transaction_Velocity'] = 1 / (group['time_diff'] + 1)
        return group
    
    df = df.groupby('user_id', group_keys=False).apply(calc_velocity, include_groups=False)
    df['Transaction_Velocity'] = df['Transaction_Velocity'].fillna(0)
    df = df.drop('time_diff', axis=1, errors='ignore')
    
    return df


def calculate_merchant_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    merchant_fraud_rate = df.groupby('merchant_type')['is_fraud'].mean().to_dict()
    
    overall_fraud_rate = df['is_fraud'].mean()
    
    df['Merchant_Risk_Score'] = df['merchant_type'].map(merchant_fraud_rate)
    df['Merchant_Risk_Score'] = df['Merchant_Risk_Score'].fillna(overall_fraud_rate)
    
    min_score = df['Merchant_Risk_Score'].min()
    max_score = df['Merchant_Risk_Score'].max()
    if max_score > min_score:
        df['Merchant_Risk_Score'] = (df['Merchant_Risk_Score'] - min_score) / (max_score - min_score)
    else:
        df['Merchant_Risk_Score'] = 0.5
    
    return df


def calculate_user_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    user_stats = df.groupby('user_id').agg({
        'transaction_amount': ['mean', 'std', 'max', 'min'],
        'balance_before': 'mean'
    }).reset_index()
    
    user_stats.columns = ['user_id', 'user_mean_amount', 'user_std_amount', 
                          'user_max_amount', 'user_min_amount', 'user_avg_balance']
    
    df = df.merge(user_stats, on='user_id', how='left')
    
    df['Amount_Normalized'] = (df['transaction_amount'] - df['user_mean_amount']) / (df['user_std_amount'] + 1)
    df['Amount_To_Balance_Ratio'] = df['transaction_amount'] / (df['balance_before'] + 1)
    df['Amount_To_Max_Ratio'] = df['transaction_amount'] / (df['user_max_amount'] + 1)
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    df['Hour'] = df['transaction_date'].dt.hour
    df['DayOfWeek'] = df['transaction_date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    channel_mapping = {'POS': 0, 'ATM': 1, 'ONLINE': 2}
    df['Channel_Encoded'] = df['channel'].map(channel_mapping).fillna(2)
    
    merchant_types = df['merchant_type'].unique()
    merchant_mapping = {m: i for i, m in enumerate(merchant_types)}
    df['Merchant_Encoded'] = df['merchant_type'].map(merchant_mapping)
    
    return df


def engineer_features(clean_filepath: str, feature_filepath: str) -> pd.DataFrame:
    print("Loading cleaned data...")
    df = pd.read_csv(clean_filepath)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    print(f"Loaded {len(df)} rows")
    
    print("\nCalculating Last 6 Month Average...")
    df = calculate_last_6_month_avg(df)
    
    print("Calculating Current Month Cumulative Sum...")
    df = calculate_current_month_cumsum(df)
    
    print("Calculating Deviation from Average...")
    df = calculate_deviation_from_avg(df)
    
    print("Calculating Rolling Standard Deviation...")
    df = calculate_rolling_std(df)
    
    print("Calculating Transaction Velocity...")
    df = calculate_transaction_velocity(df)
    
    print("Calculating Merchant Risk Score...")
    df = calculate_merchant_risk_score(df)
    
    print("Calculating User Normalized Features...")
    df = calculate_user_normalized_features(df)
    
    print("Adding Time-based Features...")
    df = add_time_features(df)
    
    print("Encoding Categorical Features...")
    df = encode_categorical_features(df)
    
    os.makedirs(os.path.dirname(feature_filepath), exist_ok=True)
    df.to_csv(feature_filepath, index=False)
    print(f"\nFeature-engineered data saved to {feature_filepath}")
    
    return df


def get_feature_columns() -> list:
    return [
        'transaction_amount', 'balance_before', 'balance_after',
        'Last_6_Month_Avg', 'Current_Month_Cumulative_Sum',
        'Deviation_From_Avg', 'Deviation_Ratio', 'Rolling_Std',
        'Transaction_Velocity', 'Merchant_Risk_Score',
        'Amount_Normalized', 'Amount_To_Balance_Ratio', 'Amount_To_Max_Ratio',
        'Hour', 'DayOfWeek', 'IsWeekend', 'IsNightTime',
        'Channel_Encoded', 'Merchant_Encoded'
    ]


if __name__ == "__main__":
    engineer_features("data/clean/transactions_clean.csv", "data/feature_engineered_data.csv")
