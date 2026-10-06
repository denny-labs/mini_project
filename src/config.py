import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# File paths
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, 'nsei_daily.csv')
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'nsei_cleaned.csv')

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)