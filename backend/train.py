import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import json
import os

from backend.features import get_feature_columns


def load_feature_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def prepare_training_data(df: pd.DataFrame, feature_columns: list):
    available_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[available_columns].copy()
    y = df['is_fraud'].copy()
    
    X = X.fillna(0)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, y, available_columns


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("\nTraining Random Forest Classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    print("Random Forest training complete!")
    
    return rf_model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    print("\nTraining XGBoost Classifier...")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train, y_train)
    print("XGBoost training complete!")
    
    return xgb_model


def get_base_model_predictions(rf_model, xgb_model, X):
    rf_proba = rf_model.predict_proba(X)[:, 1]
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    
    meta_features = np.column_stack([rf_proba, xgb_proba])
    
    return meta_features


def train_meta_learner(rf_model, xgb_model, X_train, y_train) -> LogisticRegression:
    print("\nTraining Meta-Learner (Logistic Regression)...")
    
    meta_features = get_base_model_predictions(rf_model, xgb_model, X_train)
    
    meta_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    meta_model.fit(meta_features, y_train)
    print("Meta-Learner training complete!")
    
    return meta_model


def evaluate_stacked_model(rf_model, xgb_model, meta_model, X_test, y_test):
    print("\n" + "="*50)
    print("STACKED MODEL EVALUATION")
    print("="*50)
    
    meta_features = get_base_model_predictions(rf_model, xgb_model, X_test)
    
    y_pred = meta_model.predict(meta_features)
    y_proba = meta_model.predict_proba(meta_features)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    
    return auc_score


def save_models(rf_model, xgb_model, meta_model, feature_columns: list, models_dir: str = "models"):
    os.makedirs(models_dir, exist_ok=True)
    
    rf_path = os.path.join(models_dir, "base_rf.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"Random Forest saved to {rf_path}")
    
    xgb_path = os.path.join(models_dir, "base_xgb.pkl")
    joblib.dump(xgb_model, xgb_path)
    print(f"XGBoost saved to {xgb_path}")
    
    meta_path = os.path.join(models_dir, "stacked_model.pkl")
    joblib.dump(meta_model, meta_path)
    print(f"Meta-Learner saved to {meta_path}")
    
    features_path = os.path.join(models_dir, "feature_columns.json")
    with open(features_path, 'w') as f:
        json.dump(feature_columns, f)
    print(f"Feature columns saved to {features_path}")


def train_full_pipeline(feature_data_path: str = "data/feature_engineered_data.csv"):
    print("="*60)
    print("FINANCIAL FRAUD DETECTION - MODEL TRAINING PIPELINE")
    print("="*60)
    
    print("\nLoading feature-engineered data...")
    df = load_feature_data(feature_data_path)
    print(f"Loaded {len(df)} transactions")
    
    feature_columns = get_feature_columns()
    X, y, used_columns = prepare_training_data(df, feature_columns)
    
    print(f"\nUsing {len(used_columns)} features for training")
    print(f"Fraud distribution: {y.value_counts().to_dict()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    rf_model = train_random_forest(X_train, y_train)
    
    xgb_model = train_xgboost(X_train, y_train)
    
    meta_model = train_meta_learner(rf_model, xgb_model, X_train, y_train)
    
    auc_score = evaluate_stacked_model(rf_model, xgb_model, meta_model, X_test, y_test)
    
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    save_models(rf_model, xgb_model, meta_model, used_columns)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print(f"Final Stacked Model AUC: {auc_score:.4f}")
    print("="*60)
    
    return rf_model, xgb_model, meta_model, used_columns


if __name__ == "__main__":
    train_full_pipeline()
