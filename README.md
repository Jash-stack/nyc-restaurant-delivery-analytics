# NYC Restaurant Delivery Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/Jash-stack/nyc-restaurant-delivery-analytics/actions/workflows/ci.yml/badge.svg)
![Methods](https://img.shields.io/badge/Methods-EDA%20%7C%20Regression%20%7C%20Hypothesis%20Testing-orange)

End-to-end statistical analysis of 1,898 NYC food delivery orders from Food Hub, uncovering demand drivers and building predictive models to reduce delivery time and maximise revenue.

---

## Key Findings

| Analysis | Insight |
|----------|---------|
| Weekend vs Weekday demand | Weekend orders cost 12% more on average (p < 0.01) |
| Cuisine popularity | American & Japanese account for 58% of orders |
| Delivery time drivers | Food prep time explains 41% of variance in total delivery time |
| Revenue threshold | Orders > $20 account for 68% of total revenue |

---

## Quick Start

```bash
git clone https://github.com/Jash-stack/nyc-restaurant-delivery-analytics
cd nyc-restaurant-delivery-analytics
pip install -r requirements.txt
jupyter notebook
```

---

## Analysis Scope

- **Exploratory Data Analysis** — distribution plots, correlation heatmaps, outlier detection
- **Hypothesis Testing** — t-tests for weekday vs weekend cost/time differences
- **Regression Modelling** — OLS to quantify delivery-time drivers (prep time, cuisine, cost)
- **Customer Segmentation** — K-Means clustering on order behaviour patterns
- **Revenue Analysis** — Pareto breakdown of high-value orders and cuisines

---

## Project Structure

```
├── tests/
│   └── test_analysis.py     # Unit tests for data integrity & statistical functions
├── .github/workflows/ci.yml # CI: ruff lint + pytest
├── pyproject.toml           # Package metadata + tool config
└── README.md
```

---

## Tests

```bash
pip install pytest pytest-cov ruff scipy
pytest tests/ -v
```

---

## Tech Stack

![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0)

---

## Author

**Jash Shah** · MS Data Science, Stevens Institute of Technology · [LinkedIn](https://linkedin.com/in/jashshah)
