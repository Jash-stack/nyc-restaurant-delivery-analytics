"""Exploratory data analysis utilities for NYC restaurant delivery data.

Provides data loading, cleaning, and profiling functions that prepare the
raw delivery dataset for statistical analysis and modelling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {
    "restaurant_id", "order_id", "delivery_time_min",
    "order_value_usd", "cuisine_type", "borough",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the NYC delivery dataset from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset with standard column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the delivery dataset.

    Steps applied:
    1. Drop duplicate order IDs.
    2. Remove rows with null values in key columns.
    3. Filter out physically impossible delivery times (<= 0 or > 240 min).
    4. Filter out non-positive order values.
    5. Strip and title-case string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw delivery dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with reset index.
    """
    n_start = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    df = df.dropna(subset=list(_REQUIRED_COLS))
    df = df[(df["delivery_time_min"] > 0) & (df["delivery_time_min"] <= 240)]
    df = df[df["order_value_usd"] > 0]
    for col in ["cuisine_type", "borough"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    df = df.reset_index(drop=True)
    logger.info("Cleaned data: %d -> %d rows (removed %d)", n_start, len(df), n_start - len(df))
    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compute a summary profile of the delivery dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery dataframe.

    Returns
    -------
    dict
        Keys: n_rows, n_restaurants, n_orders, boroughs, cuisine_types,
        avg_delivery_min, median_delivery_min, avg_order_value_usd.
    """
    profile = {
        "n_rows": len(df),
        "n_restaurants": df["restaurant_id"].nunique(),
        "n_orders": df["order_id"].nunique(),
        "boroughs": sorted(df["borough"].unique().tolist()),
        "cuisine_types": sorted(df["cuisine_type"].unique().tolist()),
        "avg_delivery_min": round(df["delivery_time_min"].mean(), 2),
        "median_delivery_min": round(df["delivery_time_min"].median(), 2),
        "avg_order_value_usd": round(df["order_value_usd"].mean(), 2),
    }
    logger.info("Profile: %d orders across %d restaurants", profile["n_orders"], profile["n_restaurants"])
    return profile


def delivery_time_by_group(
    df: pd.DataFrame,
    group_col: str,
    agg: str = "mean",
) -> pd.Series:
    """Aggregate delivery times by a categorical grouping column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery dataframe.
    group_col : str
        Column to group by (e.g. "borough", "cuisine_type").
    agg : str
        Aggregation function: "mean", "median", or "std".

    Returns
    -------
    pd.Series
        Aggregated delivery times indexed by group, sorted descending.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not found in dataframe")
    agg_fn = {"mean": np.mean, "median": np.median, "std": np.std}.get(agg)
    if agg_fn is None:
        raise ValueError(f"Unsupported agg={agg!r}. Choose mean, median, or std.")
    result = df.groupby(group_col)["delivery_time_min"].agg(agg_fn).sort_values(ascending=False)
    return result


def compute_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame reporting null counts and percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Any dataframe.

    Returns
    -------
    pd.DataFrame
        Columns: "null_count", "null_pct". Rows are column names.
        Only columns with at least one null are included.
    """
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    pcts = (nulls / len(df) * 100).round(2)
    return pd.DataFrame({"null_count": nulls, "null_pct": pcts})
