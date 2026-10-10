import pandas as pd
import numpy as np
import logging
from src.config import CLEANED_DATA_PATH, PROCESSED_DATA_DIR
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, 'nsei_features.csv')


def create_lag_features(df):
    """Phase 2.1: Lag returns and absolute returns."""
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)
    for lag in [1, 2, 3, 5]:
        df[f'AbsReturn_lag{lag}'] = df['Return'].abs().shift(lag)
    return df


def create_rolling_volatility(df):
    """Phase 2.2: Rolling std of returns."""
    for window in [5, 10, 20]:
        df[f'Vol_rolling_{window}'] = df['Return'].rolling(window=window).std()
    df['AbsReturn_roll_5'] = df['Return'].abs().rolling(window=5).mean()
    return df


def create_technical_indicators(df):
    """Phase 2.3: MA, RSI, MACD, ATR."""
    # Moving averages
    for window in [5, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df['Close_MA20_ratio'] = df['Close'] / df['MA_20']
    df['Close_MA50_ratio'] = df['Close'] / df['MA_50']

    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ATR(14)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    return df


def create_volume_features(df):
    """Phase 2.4: Volume-based features."""
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA_20']
    return df


def create_target(df, forward_window=5):
    """Phase 2.5: Next-day forward volatility target (no leakage)."""
    df['Target_Volatility'] = (
        df['Return'].rolling(window=forward_window).std().shift(-forward_window)
    )
    return df


def run_feature_engineering():
    logging.info("Starting Phase 2: Feature Engineering")
    df = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['Date'], index_col='Date')

    df = create_lag_features(df)
    df = create_rolling_volatility(df)
    df = create_technical_indicators(df)
    df = create_volume_features(df)
    df = create_target(df, forward_window=5)

    logging.info(f"Shape before dropping NaNs: {df.shape}")
    df = df.dropna()
    logging.info(f"Shape after dropping NaNs: {df.shape}")

    df.to_csv(FEATURES_PATH)
    logging.info(f"Phase 2 complete! Features saved to {FEATURES_PATH}")
    return df


if __name__ == "__main__":
    run_feature_engineering()