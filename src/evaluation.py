import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
    """Compute MAE, RMSE, and R²."""
    return {
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2':   float(r2_score(y_true, y_pred)),
    }


def directional_accuracy(y_true, y_pred, reference):
    """
    Directional accuracy for volatility.
    Question: does the model correctly predict whether forward volatility
    will rise or fall relative to the *current* trailing volatility?

    reference = trailing volatility known at time t (e.g., Vol_rolling_5).
    """
    pred_dir   = np.sign(np.asarray(y_pred) - np.asarray(reference))
    actual_dir = np.sign(np.asarray(y_true) - np.asarray(reference))
    return float((pred_dir == actual_dir).mean())


def regime_wise_metrics(y_true, y_pred, regimes):
    """MAE/RMSE/R² split by volatility regime (Low/Medium/High)."""
    out = {}
    for regime in pd.Series(regimes).unique():
        mask = (regimes == regime).values if hasattr(regimes, 'values') else (regimes == regime)
        out[regime] = evaluate_regression(y_true[mask], y_pred[mask])
    return pd.DataFrame(out).T