"""Statistical analysis utilities for NYC restaurant delivery data.

Provides hypothesis tests, effect-size computations, and confidence-interval
helpers used in the analysis notebooks and test suite.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def two_sample_ttest(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> dict:
    """Run a two-sample (Welch) t-test and return annotated results.

    Parameters
    ----------
    group_a, group_b : pd.Series
        Numeric samples to compare.
    alpha : float
        Significance level (default 0.05).
    equal_var : bool
        If True, use Student t-test (equal variances). Default is Welch.

    Returns
    -------
    dict
        Keys: t_stat, p_value, significant, cohens_d, mean_a, mean_b, diff.
    """
    t_stat, p_value = stats.ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=equal_var)
    d = cohens_d(group_a, group_b)
    result = {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "cohens_d": round(d, 4),
        "mean_a": round(float(group_a.mean()), 4),
        "mean_b": round(float(group_b.mean()), 4),
        "diff": round(float(group_a.mean() - group_b.mean()), 4),
    }
    logger.info("t-test: t=%.3f p=%.4f significant=%s", t_stat, p_value, result["significant"])
    return result


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute Cohen's d effect size between two groups.

    Parameters
    ----------
    group_a, group_b : pd.Series
        Numeric samples.

    Returns
    -------
    float
        Cohen's d (positive means group_a > group_b).
    """
    a, b = group_a.dropna(), group_b.dropna()
    pooled_std = np.sqrt((
        (len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2
    ) / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(
    series: pd.Series,
    statistic=np.mean,
    n_bootstrap: int = 2_000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Parameters
    ----------
    series : pd.Series
        Numeric data.
    statistic : callable
        Statistic to bootstrap (default np.mean).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level, e.g. 0.95 for 95% CI.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (lower, upper)
        Bootstrap confidence interval bounds.
    """
    rng = np.random.default_rng(random_state)
    data = series.dropna().values
    boot_stats = [
        statistic(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    lower, upper = np.percentile(boot_stats, [alpha * 100, (1 - alpha) * 100])
    return float(lower), float(upper)


def anova_delivery_by_group(df: pd.DataFrame, group_col: str) -> dict:
    """One-way ANOVA of delivery_time_min across categories of group_col.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned delivery dataframe.
    group_col : str
        Categorical column to split on (e.g. "borough", "cuisine_type").

    Returns
    -------
    dict
        Keys: f_stat, p_value, significant, groups (list of group names).
    """
    groups = [g["delivery_time_min"].dropna().values for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups for ANOVA, got {len(groups)}")
    f_stat, p_value = stats.f_oneway(*groups)
    result = {
        "f_stat": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "groups": sorted(df[group_col].unique().tolist()),
    }
    logger.info("ANOVA on %s: F=%.3f p=%.4f", group_col, f_stat, p_value)
    return result


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute pairwise correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numeric columns.
    method : str
        Correlation method: "pearson", "spearman", or "kendall".

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric columns found in dataframe")
    return numeric_df.corr(method=method)
