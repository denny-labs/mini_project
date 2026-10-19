import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# ---------- Model builders ----------

def build_xgboost(params):
    return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)


def build_lightgbm(params):
    return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)


def build_catboost(params):
    return CatBoostRegressor(**params, random_state=42, verbose=False)


# ---------- Optuna objectives ----------

def _cv_rmse(builder, params, X, y, n_splits=5, gap=5):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    for tr, va in tscv.split(X):
        model = builder(params)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        scores.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
    return float(np.mean(scores))


def xgb_objective(trial, X, y, gap=5):
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 200, 1500),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    return _cv_rmse(build_xgboost, params, X, y, gap=gap)


def lgbm_objective(trial, X, y, gap=5):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
        'num_leaves':        trial.suggest_int('num_leaves', 7, 127),
        'max_depth':         trial.suggest_int('max_depth', 3, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }
    return _cv_rmse(build_lightgbm, params, X, y, gap=gap)


def catboost_objective(trial, X, y, gap=5):
    params = {
        'iterations':         trial.suggest_int('iterations', 300, 1500),
        'depth':              trial.suggest_int('depth', 4, 10),
        'learning_rate':      trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature':trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength':    trial.suggest_float('random_strength', 0.0, 10.0),
    }
    return _cv_rmse(build_catboost, params, X, y, gap=gap)