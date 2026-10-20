import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf

# Set page config
st.set_page_config(page_title="NSEI Volatility Forecaster", layout="wide", page_icon="📈")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'nsei_features.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# ==========================================
# CACHED FUNCTIONS
# ==========================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'], index_col='Date')
    return df

@st.cache_resource
def load_models():
    meta = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl'))
    models = {}
    model_names = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
    for file_prefix, display_name in model_names.items():
        path = os.path.join(MODEL_DIR, f'{file_prefix}_model.pkl')
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
    return meta, models

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

def plotly_shap_beeswarm(shap_values, X):
    """Creates an interactive Plotly version of the SHAP Beeswarm plot."""
    feature_names = X.columns
    # Calculate mean absolute SHAP value for each feature (for sorting)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    
    fig = go.Figure()
    
    # Add a trace for each feature
    for i in sorted_idx:
        f_name = feature_names[i]
        f_vals = X.iloc[:, i].values
        
        # Normalize feature values to 0-1 for coloring (Blue=low, Red=high)
        if f_vals.max() == f_vals.min():
            colors = [0.5] * len(f_vals)
        else:
            colors = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min())
            
        fig.add_trace(go.Scatter(
            x=shap_values[:, i],
            y=[f_name] * len(f_vals),
            mode='markers',
            marker=dict(
                color=colors,
                colorscale='RdBu_r',
                size=6,
                showscale=bool(i == sorted_idx[-1]), # Show colorbar on the top feature
                colorbar=dict(title="Feature value", thickness=15)
            ),
            name=f_name,
            text=[f"Feature: {f_name}<br>Value: {v:.4f}<br>SHAP: {s:.4f}" for v, s in zip(f_vals, shap_values[:, i])],
            hoverinfo='text',
            showlegend=False
        ))
        
    fig.update_layout(
        title="SHAP Beeswarm (Interactive)",
        xaxis_title="SHAP value (impact on prediction)",
        yaxis_title="Feature",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plotly_shap_waterfall(explainer, shap_values, row_features, date_str):
    """Creates an interactive Plotly version of the SHAP Waterfall plot."""
    base_val = explainer.expected_value
    # Handle different SHAP return formats for base value
    if isinstance(base_val, (list, np.ndarray)): 
        base_val = base_val[0] if len(base_val) > 0 else 0
    
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    vals = vals[0]
    
    features = row_features.iloc[0].values
    names = row_features.columns
    
    # Sort by absolute impact
    sort_idx = np.argsort(np.abs(vals))
    
    # Calculate cumulative sums for waterfall effect
    cumulative = base_val
    measures = ['relative'] * len(vals)
    x_vals = vals[sort_idx]
    y_labels = [f"{names[i]} = {features[i]:.4f}" for i in sort_idx]
    
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(x_vals),
        y=y_labels,
        x=x_vals,
        text=[f"{v:+.4f}" for v in x_vals],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#1f77b4"}}, # Blue for pushing down
        increasing={"marker": {"color": "#d62728"}}, # Red for pushing up
    ))
    
    fig.update_layout(
        title=f"Local Explanation (Waterfall) - {date_str}",
        xaxis_title=f"SHAP Value (Base: {base_val:.4f})",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# Load data
df = load_data()
meta, models = load_models()
FEATURES = meta['features']
TARGET = meta['target']

# Recreate test split
train_end = int(len(df) * 0.80)
test_df = df.iloc[train_end:].copy()
X_test = test_df[FEATURES]

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📈 NSEI Volatility Forecaster")
st.sidebar.markdown("Forecasting 5-day forward volatility using ensemble machine learning.")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Select Model for Predictions:",
    options=list(models.keys()),
    index=list(models.keys()).index(meta['best_model'])
)
selected_model = models[selected_model_name]

y_pred = selected_model.predict(X_test)

st.sidebar.markdown(f"**Currently viewing:** {selected_model_name}")
st.sidebar.markdown(f"**Date Range:** {test_df.index.min().date()} to {test_df.index.max().date()}")

# ==========================================
# MAIN PAGE TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA Highlights", "🔮 Forecast", "⚖️ Model Comparison", "🧠 Explainability"])

# --- TAB 1: EDA (Interactive Plotly) ---
with tab1:
    st.header("Exploratory Data Analysis")
    st.markdown("Hover over the charts to see exact values and dates.")
    
    # 1. Price Trend
    st.subheader("NSEI Close Price")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='steelblue')))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='MA 20', line=dict(color='orange')))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['MA_50'], name='MA 50', line=dict(color='crimson')))
    fig_price.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_price)
    # st.plotly_chart(fig_price, use_container_width=True)

    # 2. Volatility Over Time
    st.subheader("Rolling Volatility")
    fig_vol = go.Figure()
    for col, color in zip(['Vol_rolling_5', 'Vol_rolling_10', 'Vol_rolling_20'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        fig_vol.add_trace(go.Scatter(x=df.index, y=df[col], name=col.replace('_', ' '), line=dict(color=color)))
    fig_vol.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_vol)
    # st.plotly_chart(fig_vol, use_container_width=True)


# --- TAB 2: FORECAST ---
with tab2:
    st.header(f"Forecast Viewer - {selected_model_name}")
    
    date_range = st.date_input(
        "Filter Date Range:",
        [test_df.index.min(), test_df.index.max()],
        min_value=test_df.index.min(),
        max_value=test_df.index.max()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (test_df.index >= pd.to_datetime(start_date)) & (test_df.index <= pd.to_datetime(end_date))
        
        chart_data = pd.DataFrame({
            'Actual Volatility': test_df.loc[mask, TARGET],
            'Predicted Volatility': y_pred[mask]
        }, index=test_df.index[mask])
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Actual Volatility'], name='Actual', line=dict(color='#1f77b4')))
        fig_forecast.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Predicted Volatility'], name='Predicted', line=dict(color='#d62728', dash='dash')))
        fig_forecast.update_layout(height=500, hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_forecast)
        # st.plotly_chart(fig_forecast, use_container_width=True)

# --- TAB 3: MODEL COMPARISON ---
with tab3:
    st.header("Model Comparison")
    
    comparison_path = os.path.join(BASE_DIR, 'reports', 'tables', 'model_comparison.csv')
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        
        # Display table
        # st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen').highlight_max(subset=['R²'], color='lightgreen'), use_container_width=True)
        st.dataframe(comparison_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen').highlight_max(subset=['R²'], color='lightgreen'))

        # Interactive Bar Chart
        st.subheader("Metric Comparison (Interactive)")
        
        # Melt dataframe for grouped bar chart
        df_melted = comparison_df.melt(id_vars='Model', value_vars=['MAE', 'RMSE', 'R²'], var_name='Metric', value_name='Value')
        
        fig_bar = px.bar(df_melted, x='Model', y='Value', color='Metric', barmode='group', 
                         title="Error and R² Metrics by Model", height=450)
        fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar)
        # st.plotly_chart(fig_bar, use_container_width=True)


# --- TAB 4: EXPLAINABILITY ---
# with tab4:
#     st.header("Explainability (SHAP)")
    
#     st.subheader("Global Feature Importance")
#     st.image(os.path.join(FIG_DIR, 'catboost_shap_beeswarm.png'), caption="SHAP Beeswarm Plot", use_container_width=True)
    
#     st.subheader("Local Prediction Explanation")
    
#     selected_date = st.select_slider(
#         "Select a date:",
#         options=test_df.index,
#         value=test_df.index[-1]
#     )
    
#     idx = test_df.index.get_loc(selected_date)
#     row_features = X_test.iloc[[idx]]
#     actual_val = test_df.loc[selected_date, TARGET]
#     pred_val = y_pred[idx]
    
#     col1, col2 = st.columns(2)
#     col1.metric("Actual Volatility", f"{actual_val:.4f}")
#     col2.metric("Predicted Volatility", f"{pred_val:.4f}", delta=f"{pred_val - actual_val:.4f}")
    
#     explainer = get_explainer(selected_model)
#     shap_values = explainer.shap_values(row_features)
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     exp = shap.Explanation(
#         values=shap_values[0],
#         base_values=explainer.expected_value,
#         data=row_features.iloc[0].values,
#         feature_names=row_features.columns
#     )
#     shap.waterfall_plot(exp, show=False)
#     plt.title(f"Waterfall: {selected_date.date()}")
#     st.pyplot(fig)

# --- TAB 4: EXPLAINABILITY ---
with tab4:
    st.header("Explainability (SHAP)")
    
    # We cache the explainer and shap values to prevent lag when clicking
    explainer = get_explainer(selected_model)
    
    # We only calculate SHAP values for the test set once per model
    @st.cache_data
    def get_test_shap(model_name):
        _ , _m = load_models()
        _model = _m[model_name]
        _explainer = get_explainer(_model)
        return _explainer.shap_values(X_test)
        
    shap_values_test = get_test_shap(selected_model_name)

    st.subheader("Global Feature Importance (Beeswarm)")
    st.markdown("Hover over the dots to see feature values and their exact impact. Red pushes volatility up, Blue pushes it down.")
    
    # Generate Plotly Beeswarm
    fig_beeswarm = plotly_shap_beeswarm(shap_values_test, X_test)
    st.plotly_chart(fig_beeswarm)
    # st.plotly_chart(fig_beeswarm, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Local Prediction Explanation")
    st.markdown("Select a specific date to see exactly *why* the model made its prediction.")
    
    selected_date = st.select_slider(
        "Select a date:",
        options=test_df.index,
        value=test_df.index[-1]
    )
    
    idx = test_df.index.get_loc(selected_date)
    row_features = X_test.iloc[[idx]]
    actual_val = test_df.loc[selected_date, TARGET]
    pred_val = y_pred[idx]
    
    col1, col2 = st.columns(2)
    col1.metric("Actual Volatility", f"{actual_val:.4f}")
    col2.metric("Predicted Volatility", f"{pred_val:.4f}", delta=f"{pred_val - actual_val:.4f}")
    
    # Generate Plotly Waterfall
    shap_values_row = shap_values_test[idx:idx+1]
    fig_waterfall = plotly_shap_waterfall(explainer, shap_values_row, row_features, selected_date.date())
    st.plotly_chart(fig_waterfall)
    # st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.info("💡 **How to read this:** The base value is the average volatility prediction. Red bars push the volatility higher, and blue bars push it lower. Hover over any bar to see the exact SHAP value contribution.")
