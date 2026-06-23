"""Predictive models for NYC restaurant delivery time estimation.

Implements feature engineering and scikit-learn compatible estimators for
predicting delivery time and classifying high/low value orders.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

_NUMERIC_FEATURES = ["order_value_usd", "distance_km", "hour_of_day", "day_of_week"]
_CATEGORICAL_FEATURES = ["borough", "cuisine_type"]
_TARGET = "delivery_time_min"


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from the cleaned delivery dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery dataframe with at least the required columns.

    Returns
    -------
    tuple of (X, y)
        X is a DataFrame of features; y is the delivery_time_min Series.
    """
    available_num = [c for c in _NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in _CATEGORICAL_FEATURES if c in df.columns]
    feature_cols = available_num + available_cat
    if _TARGET not in df.columns:
        raise ValueError(f"Target column {_TARGET!r} not found in dataframe")
    X = df[feature_cols].copy()
    y = df[_TARGET]
    logger.info("Feature matrix: %d rows x %d cols", len(X), len(feature_cols))
    return X, y


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric and encodes categorical features.

    Parameters
    ----------
    numeric_cols : list of str
        Numeric feature column names.
    categorical_cols : list of str
        Categorical feature column names.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready for use in a Pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def build_pipeline(model_name: str = "ridge") -> Pipeline:
    """Build a scikit-learn Pipeline for delivery time regression.

    Parameters
    ----------
    model_name : str
        One of "linear", "ridge", "random_forest", or "gradient_boosting".

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted preprocessing + estimator pipeline.
    """
    model_map = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model_name {model_name!r}. Choose from {list(model_map)}")
    available_num = _NUMERIC_FEATURES
    available_cat = _CATEGORICAL_FEATURES
    preprocessor = build_preprocessor(available_num, available_cat)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model_map[model_name]),
    ])
    logger.info("Built %s pipeline", model_name)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a fitted pipeline on held-out test data.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted sklearn Pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True target values.

    Returns
    -------
    dict
        Keys: rmse, mae, r2.
    """
    y_pred = pipeline.predict(X_test)
    metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
    }
    logger.info("Evaluation: RMSE=%.3f MAE=%.3f R2=%.3f", metrics["rmse"], metrics["mae"], metrics["r2"])
    return metrics


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict:
    """Run k-fold cross-validation and return mean/std of RMSE and R2.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted sklearn Pipeline.
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Keys: rmse_mean, rmse_std, r2_mean, r2_std.
    """
    neg_mse = cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    r2 = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)
    result = {
        "rmse_mean": round(float(np.sqrt(-neg_mse).mean()), 4),
        "rmse_std": round(float(np.sqrt(-neg_mse).std()), 4),
        "r2_mean": round(float(r2.mean()), 4),
        "r2_std": round(float(r2.std()), 4),
    }
    logger.info("%d-fold CV: RMSE=%.3f+/-%.3f R2=%.3f+/-%.3f", cv,
                result["rmse_mean"], result["rmse_std"],
                result["r2_mean"], result["r2_std"])
    return result
