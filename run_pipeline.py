import logging
from src.data_cleaning import run_cleaning_pipeline
from src.feature_engineering import run_feature_engineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    print("=== NSEI Volatility Forecasting Pipeline ===")
    print("\n--- Phase 1: Data Cleaning ---")
    df_cleaned = run_cleaning_pipeline()
    print(f"Phase 1 finished. Shape: {df_cleaned.shape}")

    print("\n--- Phase 2: Feature Engineering ---")
    df_features = run_feature_engineering()
    print(f"Phase 2 finished. Shape: {df_features.shape}")


if __name__ == "__main__":
    main()