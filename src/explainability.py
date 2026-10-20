import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure SHAP uses a non-interactive backend if run from pipeline
# import matplotlib
# matplotlib.use('Agg') 

def generate_global_shap(model, X, feature_names, model_name, output_dir):
    """Generates and saves global feature importance (bar and beeswarm)."""
    # Use TreeExplainer for tree-based models (XGBoost, LightGBM, CatBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. Bar Plot (Mean absolute impact)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, class_names=feature_names)
    plt.title(f'Global Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_shap_bar.png'), dpi=120)
    plt.close()
    
    # 2. Beeswarm Plot (Impact distribution)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Beeswarm - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_shap_beeswarm.png'), dpi=120)
    plt.close()
    
    return explainer, shap_values

def generate_local_shap(explainer, shap_values, X, indices, output_dir):
    """Generates waterfall plots for specific indices."""
    for idx in indices:
        plt.figure(figsize=(10, 6))
        # Create an Explanation object for the specific row
        exp = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X.iloc[idx].values,
            feature_names=X.columns
        )
        shap.waterfall_plot(exp, show=False)
        plt.title(f'Local Explanation for Date: {X.index[idx].date()}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_{X.index[idx].date()}.png'), dpi=120)
        plt.close()