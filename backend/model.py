import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from typing import Tuple, Dict, Any

from backend.features import get_feature_columns


class FraudDetectionModel:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None
        self.feature_columns = None
        self.is_loaded = False
        
    def load_models(self):
        try:
            rf_path = os.path.join(self.models_dir, "base_rf.pkl")
            xgb_path = os.path.join(self.models_dir, "base_xgb.pkl")
            meta_path = os.path.join(self.models_dir, "stacked_model.pkl")
            features_path = os.path.join(self.models_dir, "feature_columns.json")
            
            self.rf_model = joblib.load(rf_path)
            self.xgb_model = joblib.load(xgb_path)
            self.meta_model = joblib.load(meta_path)
            
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            
            self.is_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def prepare_transaction_features(self, user_id: str, amount: float, 
                                      merchant_type: str, channel: str,
                                      balance_before: float, 
                                      user_history: pd.DataFrame = None) -> pd.DataFrame:
        
        now = datetime.now()
        
        if user_history is not None and len(user_history) > 0:
            last_6_month_avg = user_history['transaction_amount'].tail(180).mean()
            current_month_cumsum = user_history[
                user_history['transaction_date'].dt.month == now.month
            ]['transaction_amount'].sum() + amount
            rolling_std = user_history['transaction_amount'].tail(30).std()
            user_mean_amount = user_history['transaction_amount'].mean()
            user_std_amount = user_history['transaction_amount'].std()
            user_max_amount = user_history['transaction_amount'].max()
            
            if len(user_history) > 1:
                last_trans_time = pd.to_datetime(user_history['transaction_date'].iloc[-1])
                time_diff = (now - last_trans_time).total_seconds() / 3600
                transaction_velocity = 1 / (time_diff + 1)
            else:
                transaction_velocity = 0
        else:
            last_6_month_avg = amount
            current_month_cumsum = amount
            rolling_std = 0
            user_mean_amount = amount
            user_std_amount = 0
            user_max_amount = amount
            transaction_velocity = 0
        
        merchant_risk_scores = {
            'Grocery': 0.1, 'Electronics': 0.4, 'Restaurant': 0.15,
            'Gas Station': 0.2, 'Online Shopping': 0.5, 'Travel': 0.35,
            'Entertainment': 0.25, 'Healthcare': 0.1, 'Utilities': 0.05,
            'Clothing': 0.2, 'Unknown': 0.3
        }
        merchant_risk = merchant_risk_scores.get(merchant_type, 0.3)
        
        channel_mapping = {'POS': 0, 'ATM': 1, 'ONLINE': 2}
        channel_encoded = channel_mapping.get(channel.upper(), 2)
        
        merchant_types = list(merchant_risk_scores.keys())
        merchant_encoded = merchant_types.index(merchant_type) if merchant_type in merchant_types else 10
        
        balance_after = balance_before - amount
        
        features = {
            'transaction_amount': amount,
            'balance_before': balance_before,
            'balance_after': balance_after,
            'Last_6_Month_Avg': last_6_month_avg,
            'Current_Month_Cumulative_Sum': current_month_cumsum,
            'Deviation_From_Avg': amount - last_6_month_avg,
            'Deviation_Ratio': amount / (last_6_month_avg + 1),
            'Rolling_Std': rolling_std if not np.isnan(rolling_std) else 0,
            'Transaction_Velocity': transaction_velocity,
            'Merchant_Risk_Score': merchant_risk,
            'Amount_Normalized': (amount - user_mean_amount) / (user_std_amount + 1),
            'Amount_To_Balance_Ratio': amount / (balance_before + 1),
            'Amount_To_Max_Ratio': amount / (user_max_amount + 1),
            'Hour': now.hour,
            'DayOfWeek': now.weekday(),
            'IsWeekend': 1 if now.weekday() >= 5 else 0,
            'IsNightTime': 1 if now.hour >= 22 or now.hour <= 6 else 0,
            'Channel_Encoded': channel_encoded,
            'Merchant_Encoded': merchant_encoded
        }
        
        feature_df = pd.DataFrame([features])
        
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[self.feature_columns]
        
        return feature_df
    
    def predict_fraud_probability(self, features: pd.DataFrame) -> float:
        if not self.is_loaded:
            if not self.load_models():
                return 0.5
        
        rf_proba = self.rf_model.predict_proba(features)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(features)[:, 1]
        
        meta_features = np.column_stack([rf_proba, xgb_proba])
        
        fraud_proba = self.meta_model.predict_proba(meta_features)[:, 1][0]
        
        return float(fraud_proba)
    
    def calculate_user_limits(self, user_id: str, balance: float, 
                              fraud_history: float = 0.0) -> Dict[str, float]:
        base_limit = balance * 0.3
        
        risk_factor = 1 - fraud_history
        leverage = base_limit * 0.5 * risk_factor
        
        total_limit = base_limit + leverage
        
        return {
            'base_limit': round(base_limit, 2),
            'leverage': round(leverage, 2),
            'total_limit': round(total_limit, 2)
        }
    
    def evaluate_transaction(self, user_id: str, amount: float, 
                            merchant_type: str, channel: str,
                            balance: float, 
                            user_history: pd.DataFrame = None) -> Dict[str, Any]:
        
        features = self.prepare_transaction_features(
            user_id, amount, merchant_type, channel, balance, user_history
        )
        
        fraud_probability = self.predict_fraud_probability(features)
        
        avg_fraud_rate = 0.1
        if user_history is not None and 'is_fraud' in user_history.columns:
            avg_fraud_rate = user_history['is_fraud'].mean() if len(user_history) > 0 else 0.1
        
        limits = self.calculate_user_limits(user_id, balance, avg_fraud_rate)
        
        if fraud_probability >= 0.8:
            status = "REJECTED"
            risk_level = "HIGH_RISK"
            message = "Transaction automatically rejected due to high fraud probability."
        elif fraud_probability >= 0.5 or amount > limits['total_limit']:
            status = "SUSPICIOUS"
            risk_level = "MEDIUM_RISK"
            if amount > limits['total_limit']:
                message = f"Transaction exceeds your ML-predicted limit (${limits['total_limit']:.2f})."
            else:
                message = "Transaction flagged as suspicious. Please review."
        else:
            status = "APPROVED"
            risk_level = "LOW_RISK"
            message = "Transaction approved."
        
        return {
            'user_id': user_id,
            'amount': amount,
            'merchant_type': merchant_type,
            'channel': channel,
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': risk_level,
            'status': status,
            'message': message,
            'limits': limits,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def log_transaction(transaction_result: Dict[str, Any], 
                   final_status: str,
                   history_path: str = "data/transactions_history.csv"):
    
    log_entry = {
        'timestamp': transaction_result['timestamp'],
        'user_id': transaction_result['user_id'],
        'amount': transaction_result['amount'],
        'merchant': transaction_result['merchant_type'],
        'channel': transaction_result['channel'],
        'status': final_status,
        'fraud_prob': transaction_result['fraud_probability'],
        'risk_level': transaction_result['risk_level']
    }
    
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    
    df.to_csv(history_path, index=False)
    
    return log_entry


def get_user_transaction_history(user_id: str, 
                                 history_path: str = "data/transactions_history.csv") -> pd.DataFrame:
    if not os.path.exists(history_path):
        return pd.DataFrame()
    
    df = pd.read_csv(history_path)
    user_history = df[df['user_id'] == user_id].copy()
    
    if 'timestamp' in user_history.columns:
        user_history['transaction_date'] = pd.to_datetime(user_history['timestamp'])
        user_history['transaction_amount'] = user_history['amount']
        user_history['is_fraud'] = (user_history['status'] == 'REJECTED').astype(int)
    
    return user_history


fraud_model = FraudDetectionModel()


if __name__ == "__main__":
    model = FraudDetectionModel()
    model.load_models()
    
    result = model.evaluate_transaction(
        user_id="USR001",
        amount=500.0,
        merchant_type="Electronics",
        channel="Online",
        balance=25000.0
    )
    
    print("\nTransaction Evaluation Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
