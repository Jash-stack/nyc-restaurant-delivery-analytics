"""Unit tests for NYC restaurant delivery analysis."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats


@pytest.fixture
def sample_orders():
    """Minimal synthetic order dataset matching the Food Hub schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "restaurant_name": np.random.choice(["A", "B", "C"], n),
        "cuisine_type": np.random.choice(["American", "Japanese", "Italian"], n),
        "cost_of_the_order": np.random.uniform(5, 50, n).round(2),
        "day_of_the_week": np.random.choice(["Weekday", "Weekend"], n),
        "rating": np.random.choice([np.nan, 3, 4, 5], n),
        "food_preparation_time": np.random.randint(10, 45, n),
        "delivery_time": np.random.randint(15, 60, n),
    })


class TestDataIntegrity:
    def test_no_negative_costs(self, sample_orders):
        assert (sample_orders["cost_of_the_order"] >= 0).all()

    def test_order_ids_unique(self, sample_orders):
        assert sample_orders["order_id"].is_unique

    def test_day_of_week_values(self, sample_orders):
        valid = {"Weekday", "Weekend"}
        assert set(sample_orders["day_of_the_week"].unique()).issubset(valid)

    def test_delivery_time_positive(self, sample_orders):
        assert (sample_orders["delivery_time"] > 0).all()


class TestStatisticalAnalysis:
    def test_mean_cost_reasonable(self, sample_orders):
        mean_cost = sample_orders["cost_of_the_order"].mean()
        assert 5 <= mean_cost <= 50

    def test_ttest_weekday_vs_weekend(self, sample_orders):
        weekday = sample_orders[sample_orders["day_of_the_week"] == "Weekday"]["cost_of_the_order"]
        weekend = sample_orders[sample_orders["day_of_the_week"] == "Weekend"]["cost_of_the_order"]
        _, p_value = stats.ttest_ind(weekday, weekend)
        assert 0 <= p_value <= 1

    def test_total_delivery_time(self, sample_orders):
        sample_orders["total_time"] = (
            sample_orders["food_preparation_time"] + sample_orders["delivery_time"]
        )
        assert (sample_orders["total_time"] > 0).all()


class TestFeatureEngineering:
    def test_revenue_above_threshold(self, sample_orders):
        high_value = sample_orders[sample_orders["cost_of_the_order"] > 20]
        assert len(high_value) > 0

    def test_cuisine_distribution(self, sample_orders):
        counts = sample_orders["cuisine_type"].value_counts()
        assert len(counts) == 3
        assert counts.sum() == len(sample_orders)
