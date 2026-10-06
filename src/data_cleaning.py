import pandas as pd
import numpy as np
import logging
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_data(filepath):
#     """Phase 1.1 & 1.2: Load data and set datetime index."""
#     logging.info("Loading raw data...")
#     # Based on the screenshot, we skip the Ticker row and the empty header row
#     df = pd.read_csv(filepath, skiprows=[0, 1])
    
#     # The first column is named 'Price' but contains the dates
#     df = df.rename(columns={'Price': 'Date'})
    
#     # Convert to datetime and set as index
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.set_index('Date')
    
#     # Sort chronologically
#     df = df.sort_index()
    
#     # Ensure all OHLCV columns are numeric
#     numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
#     for col in numeric_cols:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
        
#     logging.info(f"Data loaded successfully. Shape: {df.shape}")
#     return df

def load_data(filepath):
    """Phase 1.1 & 1.2: Load data and set datetime index."""
    logging.info("Loading raw data...")
    # Skip row 1 (Ticker) and row 2 (Date,,,,) and use row 0 as the header
    df = pd.read_csv(filepath, skiprows=[1, 2])
    
    # The first column is named 'Price' but contains the dates
    df = df.rename(columns={'Price': 'Date'})
    
    # Convert to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Sort chronologically
    df = df.sort_index()
    
    # Ensure all OHLCV columns are numeric
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def check_duplicates_and_missing(df):
    """Phase 1.3: Check duplicates and missing values."""
    duplicates = df.index.duplicated().sum()
    logging.info(f"Duplicate dates found: {duplicates}")
    if duplicates > 0:
        df = df[~df.index.duplicated(keep='first')]
        logging.info("Dropped duplicate dates.")

    missing = df.isnull().sum()
    logging.info(f"Missing values per column:\n{missing}")
    
    if missing.sum() > 0:
        # Forward fill is standard for time series market data
        df = df.ffill()
        logging.info("Forward-filled missing values.")
        
    return df

def validate_ohlc_logic(df):
    """Phase 1.4: Validate OHLC logic."""
    logging.info("Validating OHLC logic...")
    # Check that Low <= Open, Close <= High for every row
    invalid_rows = df[
        (df['Low'] > df['Open']) | 
        (df['Low'] > df['Close']) | 
        (df['High'] < df['Open']) | 
        (df['High'] < df['Close']) |
        (df['High'] < df['Low'])
    ]
    
    if not invalid_rows.empty:
        logging.warning(f"Found {len(invalid_rows)} rows violating OHLC logic. Removing them.")
        df = df.drop(invalid_rows.index)
    else:
        logging.info("All OHLC logic checks passed.")
        
    return df

def handle_outliers(df):
    """Phase 1.5: Inspect extreme values."""
    logging.info("Inspecting outliers in returns and volume...")
    # For Phase 1, we'll just document them. 
    # We don't drop them yet because market shocks (e.g., COVID crash) are genuine data.
    # We'll calculate them later after returns are created.
    return df

def create_base_derived_columns(df):
    """Phase 1.6: Create base derived columns."""
    logging.info("Creating base derived columns...")
    # Daily log returns
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    # High-Low Range
    df['High_Low_Range'] = df['High'] - df['Low']
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Drop the first row since Return and Volume_Change will be NaN
    df = df.dropna()
    
    return df

def run_cleaning_pipeline():
    """Main execution function for Phase 1."""
    logging.info("Starting Phase 1: Data Setup & Cleaning")
    
    # 1.1 & 1.2 Load and sort
    df = load_data(RAW_DATA_PATH)
    
    # 1.3 Check duplicates and missing
    df = check_duplicates_and_missing(df)
    
    # 1.4 Validate OHLC
    df = validate_ohlc_logic(df)
    
    # 1.5 Handle outliers (just logging for now)
    df = handle_outliers(df)
    
    # 1.6 Create base derived columns
    df = create_base_derived_columns(df)
    
    # Save the cleaned dataset
    df.to_csv(CLEANED_DATA_PATH)
    logging.info(f"Phase 1 Complete! Cleaned data saved to {CLEANED_DATA_PATH}")
    logging.info(f"Final Dataset Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    run_cleaning_pipeline()
