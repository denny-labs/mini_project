# NSEI Volatility Forecasting

Forecasting stock-market volatility for the NSEI (Nifty 50) index using time-series features and gradient-boosting models (XGBoost, LightGBM, CatBoost), with explainability (SHAP) and an interactive dashboard.

## Project structure
- `data/` – raw and processed datasets
- `notebooks/` – EDA, feature engineering, modeling, explainability
- `src/` – reusable Python modules
- `app/` – dashboard and API
- `models/` – trained model artifacts
- `reports/` – figures and result tables

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order (01 → 05).
3. Launch dashboard: `streamlit run app/dashboard.py`

## Base paper
“Forecasting Stock Market Volatility Using XGBoost: A Time Series Analysis” (IEEE, 2024).