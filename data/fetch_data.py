import yfinance as yf

# Download historical NIFTY 50 data
nsei_data = yf.download("^NSEI", start="2020-01-01", end="2026-10-01")

# Export to a CSV dataset for your project
nsei_data.to_csv("nsei_historical_data.csv")
print("Dataset downloaded successfully!")
