# 📈 NSEI Stock Market Volatility Forecasting

## 📌 Project Overview
Forecasting stock market volatility is crucial for risk management, option pricing, and portfolio allocation. This project predicts the **5-day forward volatility** of the National Stock Exchange of India (NSEI) using advanced ensemble machine learning techniques. 

This project goes beyond the base research paper (*"Forecasting Stock Market Volatility Using XGBoost"*) by implementing a **leakage-free time-series split**, extending the model comparison to LightGBM and CatBoost, and providing full interpretability using **SHAP (SHapley Additive exPlanations)**.

## 🎯 Key Features
- **Target Engineering:** Predicts 5-day *forward* volatility (not just next-day close price), ensuring a 5-day gap during train/test splits to prevent data leakage.
- **Advanced Feature Engineering:** Integrates OHLCV data with technical indicators (RSI, MACD, ATR) and rolling volatility windows.
- **Ensemble Modeling:** Compares XGBoost, LightGBM, and CatBoost against linear baselines using rigorous `TimeSeriesSplit` cross-validation and Optuna hyperparameter tuning.
- **Explainability:** Uses SHAP to break open the "black box," showing exactly which features drive volatility spikes (e.g., volatility clustering).
- **Interactive Dashboard:** A fully interactive Streamlit app with Plotly charts for EDA, forecasting, model comparison, and local SHAP explanations.

## 📂 Repository Structure
```text
nsei-volatility-forecasting/
│
├── data/
│   ├── raw/                  # Original dataset (nsei_daily.csv)
│   └── processed/            # Cleaned data and engineered features
│
├── notebooks/                # Step-by-step Jupyter notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_explainability.ipynb
│
├── src/                      # Production-ready Python scripts
│   ├── config.py             # Paths and constants
│   ├── data_cleaning.py      # Phase 1 pipeline
│   ├── feature_engineering.py# Phase 2 pipeline
│   ├── models.py             # Model builders and Optuna objectives
│   ├── evaluation.py         # Metric calculations
│   └── explainability.py     # SHAP generation
│
├── app/
│   └── dashboard.py          # Streamlit interactive dashboard
│
├── models/                   # Saved .pkl models and metadata
├── reports/                  # Generated figures and comparison tables
├── run_pipeline.py           # End-to-end execution script
├── requirements.txt          # Dependencies
└── README.md

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebooks in order (01 → 05).
3. Launch dashboard: `streamlit run app/dashboard.py`

## Base paper
“Forecasting Stock Market Volatility Using XGBoost: A Time Series Analysis” (IEEE, 2024).