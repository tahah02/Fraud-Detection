from backend.data_simulator import generate_raw_transactions
from backend.data_cleaning import clean_data
from backend.features import engineer_features
from backend.train import train_full_pipeline
from backend.utils import ensure_directories


def run_complete_pipeline():
    print("="*70)
    print("FINANCIAL FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    print("="*70)
    
    print("\n[Step 0] Creating directories...")
    ensure_directories()
    
    print("\n" + "="*70)
    print("[Step 1] GENERATING RAW DATA")
    print("="*70)
    generate_raw_transactions(
        num_transactions=10000,
        fraud_rate=0.05,
        output_path="data/raw/transactions_raw.csv"
    )
    
    print("\n" + "="*70)
    print("[Step 2] CLEANING DATA")
    print("="*70)
    clean_data(
        raw_filepath="data/raw/transactions_raw.csv",
        clean_filepath="data/clean/transactions_clean.csv"
    )
    
    print("\n" + "="*70)
    print("[Step 3] FEATURE ENGINEERING")
    print("="*70)
    engineer_features(
        clean_filepath="data/clean/transactions_clean.csv",
        feature_filepath="data/feature_engineered_data.csv"
    )
    
    print("\n" + "="*70)
    print("[Step 4] MODEL TRAINING")
    print("="*70)
    train_full_pipeline(
        feature_data_path="data/feature_engineered_data.csv"
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - data/raw/transactions_raw.csv (raw messy data)")
    print("  - data/clean/transactions_clean.csv (cleaned data)")
    print("  - data/feature_engineered_data.csv (feature-engineered data)")
    print("  - models/base_rf.pkl (Random Forest model)")
    print("  - models/base_xgb.pkl (XGBoost model)")
    print("  - models/stacked_model.pkl (Meta-learner model)")
    print("  - models/feature_columns.json (Feature column list)")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    run_complete_pipeline()
