# TODO — Future Experiment Ideas

Ideas discussed during planning. On hold for now.

---

## Scenario A: Feature Ablation Study

Train all 3 models (MLP, KNN, XGBoost) on 3 feature subsets:
- **34 features** (all) — baseline
- **15 features** (top by mutual information with target)
- **5 features** (minimal)

Output: RMSE vs feature count table + line chart. Measures how much feature engineering matters.

---

## Scenario B: Ensemble Method

Simple average of MLP + KNN + XGB predictions. Compare ensemble RMSE vs best individual model. Almost always beats individuals for regression.

Output: ensemble metrics table, comparison bar chart vs 3 individual models.

---

## Scenario C: Regime Analysis

Split test set by volatility level (low/medium/high based on ATR percentile). Compare model RMSE in each regime. Tests whether models handle calm vs turbulent markets differently.

Output: RMSE-by-regime table, grouped bar chart.

---

## Scenario D: Lookahead Sweep

Retrain XGBoost with different prediction horizons:
- 1-min ahead (current)
- 5-min ahead
- 15-min ahead
- 1-hour ahead

Output: RMSE decay curve vs lookahead. Shows how fast predictive power degrades.

---

*Generated during PLAN.md discussion, 2026-06-12.*
