import logging
from src.data_cleaning import run_cleaning_pipeline

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("=== NSEI Volatility Forecasting Pipeline ===")
    
    # Phase 1: Data Setup & Cleaning
    print("\n--- Running Phase 1: Data Cleaning ---")
    df_cleaned = run_cleaning_pipeline()
    print(f"Phase 1 finished. Data shape: {df_cleaned.shape}")

if __name__ == "__main__":
    main()