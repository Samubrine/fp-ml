# FP-ML: Notebook Implementation Specification

Target: forex_forecasting.ipynb | Language: English | Models: MLP+KNN+XGBoost | Sections: 7 | Estimated cells: ~70

## Global Dependencies

Shared imports: pandas, numpy, torch, torch.nn, torch.utils.data, sklearn.preprocessing (StandardScaler), sklearn.decomposition (PCA), sklearn.cluster (KMeans), sklearn.metrics, sklearn.feature_selection (mutual_info_regression), sklearn.model_selection (ParameterGrid), xgboost, matplotlib, seaborn, scipy.stats, statsmodels.tsa.stattools (adfuller), statsmodels.graphics.tsaplots (plot_acf), mplfinance, warnings, os, json, pickle, time, copy, itertools.

Shared data files: data/processed/USDCHF_1min_2020_2026.csv (~119MB, 2,319,766 rows), outputs/preprocessed/data.pt (301MB, preprocessed tensors), outputs/plots/ directory for all saved figures.

Shared variables: feature_cols (list of 34 feature name strings), chronological split dates (train_cutoff = 2025-01-01, val_cutoff = 2025-09-01), random seed = 42, device string (cuda or cpu).

Shared model outputs: outputs/models/mlp_v2.pt, outputs/models/knn_v2.pkl, outputs/models/xgboost_v2.json, evals dict (per-model metrics), preds_dict (model_name → y_pred arrays).

GPU/CPU fallback: ROCm 7.2 on AMD RX 9060 XT for GPU ops; fallback to CPU if torch.cuda.is_available() is False. XGBoost 3.1.1 (ROCm/xgboost fork) runs on AMD GPU via tree_method='hist', device='cuda'. KNN distance computation on GPU via torch.cdist with CPU fallback.

---

## Section 1: Project Description

### Block 1.1: Section Header — Project Identity
- Type: section_header
- Purpose: Orient reader — this is a group academic project on USD/CHF forex forecasting.
- Input: None
- Output: Markdown heading "1. Project Description" with subheading containing group member names, course/context, and GitHub repo link.
- Visualization Spec: None
- Narrative Content: None (structural only)
- Dependencies: None

### Block 1.2: Project Motivation — Why Forecast Forex?
- Type: markdown_narrative
- Purpose: Reader understands WHY this problem matters — both academically (model comparison under noise) and practically (forex is the world's largest financial market, $7.5T/day volume).
- Input: None
- Output: Narrative text establishing: forex market scale, difficulty of prediction (efficient market hypothesis, near-random-walk behavior), academic value of benchmarking diverse model architectures on real high-frequency data.
- Visualization Spec: None
- Narrative Content: Cover: (a) forex as largest liquid market, (b) EMH — prices should be unpredictable, (c) academic question: can any model extract signal from noise at 1-minute granularity? (d) practical relevance: algorithmic trading, market making, risk management. Do NOT introduce models yet — just the problem space.
- Dependencies: Block 1.1

### Block 1.3: What Are We Predicting? — Target Definition
- Type: markdown_narrative
- Purpose: Reader understands the prediction target is 1-step-ahead log-return, NOT absolute price — and why this choice is critical.
- Input: None
- Output: Narrative defining the target variable: y_t = ln(close_{t+1} / close_t). Explanation of why log-return (stationarity) over absolute price (non-stationary, regime-dependent). One equation block: y_t = ln(P_{t+1} / P_t).
- Visualization Spec: None
- Narrative Content: Explain: (a) absolute price non-stationary across 6 years — trends, regime shifts destroy model generalization, (b) log-return approximately stationary (ADF test p < 0.001), (c) log-return is additive over time, (d) this is a REGRESSION task (predict magnitude + direction), not classification. Reference v1 failure: MLP R² = -3.26 on absolute price.
- Dependencies: Block 1.2

### Block 1.4: Dataset Origin and Structure
- Type: markdown_narrative
- Purpose: Reader knows exactly what data is used, where it came from, its dimensions, and its schema.
- Input: None
- Output: Narrative describing: source (histdata.com), file path (data/processed/USDCHF_1min_2020_2026.csv), row count (2,319,766), date range (2020-01-01 → 2026-05-29), columns (datetime, open, high, low, close, volume=0, tick_volume, spread), file size (~119MB). Notes volume column is always 0 in forex data and will be dropped.
- Visualization Spec: None
- Narrative Content: State: (a) free historical forex data from histdata.com, (b) 1-minute OHLC bars, (c) 2.3M+ rows spanning ~6 years, (d) USD/CHF pair — one of the major forex pairs, (e) note that tick_volume and spread columns exist but are not used as features.
- Dependencies: Block 1.1

### Block 1.5: Time Series Characteristics — Regime Shifts
- Type: markdown_narrative
- Purpose: Reader understands the non-stationary nature of the raw price series across the 6-year window — bullish and bearish regimes.
- Input: None
- Output: Narrative describing 4 distinct price regimes: bullish 2020-2021 (USD strength vs CHF), sideways 2022-2024 (consolidation), bullish early 2025 (breakout), bearish late 2025-2026 (decline). Explains why this motivates log-return transformation and chronological (not random) splitting.
- Visualization Spec: None
- Narrative Content: Describe each regime qualitatively. Connect to Block 1.3: these regime shifts are WHY absolute price fails and log-return is necessary. Connect to chronological split: chronological split respects these temporal boundaries.
- Dependencies: Block 1.3, Block 1.4

### Block 1.6: Feature Engineering — 34 Inputs
- Type: markdown_narrative
- Purpose: Reader understands the breadth of engineered features before seeing any code — what categories exist and why each matters.
- Input: None
- Output: Narrative enumerating 4 feature families: (a) 8 lag features — close_lag_1..60, captures autocorrelation, (b) 16 rolling statistics — close_roll_{mean,std,min,max}_{5,10,30,60}, captures local volatility and trend, (c) 4 price-derived — log_return, pct_change, hl_spread, oc_range, captures intra-bar dynamics, (d) 6 technical indicators — rsi_14, macd_hist, bb_position_20, bb_width_20, atr_14, captures momentum and volatility regimes. Total: 34 features, each row is a 1-minute observation.
- Visualization Spec: None
- Narrative Content: For each family, explain in 1-2 sentences what it measures and why it might help predict next-minute movement. No formulas needed here — save those for the preprocessing section.
- Dependencies: Block 1.3

### Block 1.7: Experimental Design — Chronological Split
- Type: markdown_narrative
- Purpose: Reader understands the train/val/test split boundaries and WHY chronological (no shuffle) is mandatory for time series.
- Input: None
- Output: Narrative with split table: Train (< 2025-01-01, ~1.8M rows), Validation (2025-01-01 → 2025-09-01, ~247K rows), Test (≥ 2025-09-01, ~276K rows). Explanation of look-ahead bias if shuffled. Connection to real-world deployment: models are tested on truly unseen future data.
- Visualization Spec: None
- Narrative Content: Define look-ahead bias. Explain why sklearn's default train_test_split(shuffle=True) is catastrophic for time series. State the exact cutoff dates. Note that these boundaries align with natural regime boundaries (sideways → bullish in early 2025, bullish → bearish in late 2025).
- Dependencies: Block 1.4, Block 1.5

### Block 1.8: Three Models — Architecture Overview
- Type: markdown_narrative
- Purpose: Reader knows WHAT three models are compared and WHY this specific trio was chosen — different learning paradigms (neural, instance-based, tree ensemble).
- Input: None
- Output: Narrative introducing: (1) MLP — deep neural network, 34→1024→512→256→128→1, GPU-accelerated, non-linear function approximator, (2) KNN — k-nearest neighbors with k=50, GPU batched distance computation, non-parametric, (3) XGBoost — gradient-boosted trees, 65 trees, GPU histogram method (ROCm), state-of-the-art tabular model. One sentence per model on what class of algorithm it represents.
- Visualization Spec: None
- Narrative Content: Frame as a comparison across three paradigms: deep learning, lazy learning, and gradient boosting. State that all three predict the SAME log-return target on the SAME train/val/test split using the SAME 34 features and StandardScaler — the comparison is apples-to-apples.
- Dependencies: Block 1.3, Block 1.6, Block 1.7

### Block 1.9: Evaluation Framework — Five Metrics
- Type: markdown_narrative
- Purpose: Reader knows exactly how model performance is measured — 5 metrics covering error magnitude, explained variance, relative error, directional accuracy, and classification quality.
- Input: None
- Output: Narrative defining: RMSE (root mean squared error, penalizes large errors), MAE (mean absolute error, robust), R² (coefficient of determination, how much variance explained vs naive mean), MAPE (mean absolute percentage error, scale-independent), DirAcc (directional accuracy — % of times predicted sign matches actual sign). Equation for R²: R² = 1 - SS_res / SS_tot. Note: R² can be negative — model worse than always predicting the mean.
- Visualization Spec: None
- Narrative Content: Explain each metric's interpretation. Emphasize: (a) in noisy forex data, R² close to 0 is actually GOOD — random walk R² ≈ 0, negative R² means model is worse than guessing the mean, (b) DirAcc ~50% is random guessing, >50% is signal, (c) MAPE < 0.01% means predictions are extremely close to actual log-returns.
- Dependencies: Block 1.3

### Block 1.10: Three Experimental Scenarios
- Type: markdown_narrative
- Purpose: Reader understands the notebook evaluates models across three complementary lenses: regression performance, directional classification, and market regime analysis.
- Input: None
- Output: Narrative describing: Scenario 1 — regression evaluation (RMSE, MAE, R², MAPE; residual analysis; actual-vs-predicted plots), Scenario 2 — directional classification (discretize predictions to Up/Down; confusion matrix; precision/recall/F1 per direction), Scenario 3 — market regime robustness (K-Means clustering on features to identify volatility/trend regimes; evaluate if model errors concentrate in specific regimes; PCA visualization colored by regime and error).
- Visualization Spec: None
- Narrative Content: Frame each scenario as answering a different question: (1) "How well does it predict?" (2) "Does it get the direction right?" (3) "Does it work in all market conditions?" This foreshadows the notebook's evaluation structure.
- Dependencies: Block 1.8, Block 1.9

### Block 1.11: Project Deliverables and Reproducibility
- Type: markdown_narrative
- Purpose: Reader knows what artifacts exist, where they are, and how to reproduce results.
- Input: None
- Output: Narrative listing: (a) GitHub repo Samubrine/fp-ml, (b) main notebook forex_forecasting.ipynb (46 cells, Run All = full pipeline), (c) preprocessed data outputs/preprocessed/data.pt (301MB), (d) trained models mlp_v2.pt, knn_v2.pkl, xgboost_v2.json, (e) results JSONs and comparison chart outputs/plots/v2_comparison.png, (f) standalone training scripts in src/. Notes total runtime ~30 minutes and hardware (AMD RX 9060 XT, ROCm 7.2).
- Visualization Spec: None
- Narrative Content: Conclude by stating the notebook is self-contained — all preprocessing, training, evaluation, and visualization happens inline. No external data dependencies beyond the CSV. Full reproducibility instructions in README/PLAN2.md.
- Dependencies: Block 1.1, Block 1.4

---

## Section 2: Exploratory Data Analysis

### Block 2.1: Load Raw Data
- Type: code_output
- Purpose: Reader learns the CSV file's path, read mechanism, and row count.
- Input: CSV file at data/processed/USDCHF_1min_2020_2026.csv
- Output: DataFrame df with ~2.3M rows, columns: datetime, open, high, low, close, volume, tick_volume, spread
- Visualization Spec: N/A
- Narrative Content: N/A
- Dependencies: None

### Block 2.2: Data Structure Overview
- Type: table
- Purpose: Reader sees the schema at a glance — column names, dtypes, example values, and non-null counts in a single spec table.
- Input: df
- Output: Rendered table data_spec with columns: Column Name, Dtype, Non-Null Count, 3 Example Values
- Visualization Spec: N/A
- Narrative Content: N/A
- Dependencies: Block 2.1

### Block 2.3: What the Columns Mean
- Type: markdown_narrative
- Purpose: Reader understands the domain meaning of each column (OHLC bar, tick volume vs real volume, spread) and why this is forex tick data.
- Input: Column names from Block 2.2
- Output: N/A (inline markdown)
- Narrative Content: Explain OHLCV bar terminology. Define tick_volume as number of price updates in that minute (not monetary volume). Define spread as bid-ask spread in pips at bar close. Clarify that volume is near-zero because forex has no central exchange — this is a data quirk, not a bug. Note 1-minute granularity. Mention the 2020–2026 period (~6 years).
- Dependencies: Block 2.2

### Block 2.4: Missing Value Audit
- Type: table
- Purpose: Reader sees exactly where and how many values are missing, both as absolute counts and percentages, to decide on imputation or drop strategies later.
- Input: df
- Output: Rendered table missing_report with columns: Column, Missing Count, Missing %, Note (e.g., "market close gaps")
- Visualization Spec: N/A
- Narrative Content: N/A
- Dependencies: Block 2.1

### Block 2.5: Missing Value Patterns
- Type: markdown_narrative
- Purpose: Reader understands that missing values are not random — they cluster at weekends and market close hours, which matters for time-series imputation.
- Input: missing_report from Block 2.4, df
- Output: N/A (inline markdown)
- Narrative Content: Explain the forex market 24/5 schedule. Missing rows correspond to weekends (Saturday–Sunday UTC) and daily broker reset windows. This is structurally missing data, not random. Implication: forward-fill or time-aware imputation is appropriate; mean imputation is not.
- Dependencies: Block 2.4

### Block 2.6: Descriptive Statistics
- Type: table
- Purpose: Reader sees the central tendency, dispersion, and range of every numeric column to spot outliers and understand typical magnitudes.
- Input: df
- Output: Rendered table desc_stats with rows per column, columns: count, mean, std, min, 25%, 50%, 75%, max
- Visualization Spec: N/A
- Narrative Content: N/A
- Dependencies: Block 2.1

### Block 2.7: Interpreting the Stats
- Type: markdown_narrative
- Purpose: Reader connects raw statistics to trading intuition — typical daily ranges, spread costs, tick frequency.
- Input: desc_stats from Block 2.6
- Output: N/A (inline markdown)
- Narrative Content: Highlight: mean close for USD/CHF in the 0.90–1.00 range (USD/CHF typical levels), std captures multi-year dispersion. Volume mean near zero confirms tick-data quirk. tick_volume ~10–50 per minute indicates typical activity. Spread ~0–2 pips — note max spread spikes during news/rollover. Remind reader that min/max of close are the multi-year extremes.
- Dependencies: Block 2.6

### Block 2.8: Full-Period Price Series
- Type: visualization
- Purpose: Reader sees the entire 6-year price trajectory to grasp the USD/CHF macro trend (COVID crash, recovery, range-bound years).
- Input: df['datetime'], df['close']
- Output: Displayed figure
- Visualization Spec: Line plot, x = datetime (2020-01 to 2026), y = close price, single dark blue line, light gray background, title "USD/CHF Close Price — Full Period (2020–2026)", y-axis labeled "Price (CHF per USD)". Optionally shade major events (COVID March 2020, Ukraine Feb 2022) with vertical annotation bands.
- Dependencies: Block 2.1

### Block 2.9: One-Year Zoom (2025)
- Type: visualization
- Purpose: Reader inspects a recent year at higher resolution to see intra-year trends, volatility clusters, and the granularity of 1-minute data.
- Input: df['datetime'] filtered to 2025, df['close'] filtered to 2025
- Output: Displayed figure
- Visualization Spec: Line plot, x = datetime (2025-01-01 to 2025-12-31), y = close price, single dark blue line, title "USD/CHF Close Price — 2025 Zoom", y-axis "Price (CHF per USD)". No smoothing — raw 1-minute line shows true density.
- Dependencies: Block 2.8

### Block 2.10: Zoom Interpretation
- Type: markdown_narrative
- Purpose: Reader learns to read the 2025 zoom — visible weekend gaps, intraday oscillation, and volatility regime shifts.
- Input: Visualizations from Blocks 2.8 and 2.9
- Output: N/A (inline markdown)
- Narrative Content: Point out: (1) vertical gaps every weekend where no data exists, (2) intraday price movement is clearly visible at 1-min granularity, (3) volatility is not constant — there are calm periods and turbulent clusters. Tie this to why a model must handle non-stationarity and irregular sampling.
- Dependencies: Block 2.9

### Block 2.11: Daily OHLC Candlestick Sample
- Type: visualization
- Purpose: Reader sees the classic candlestick representation and understands how OHLC bars aggregate intra-minute data into tradeable signals.
- Input: df resampled to daily (df_daily with columns: datetime, open, high, low, close)
- Output: Displayed figure
- Visualization Spec: Candlestick chart (or OHLC bar chart), x = datetime (subset: last 90 trading days for legibility), green candles for close > open, red for close < open, title "USD/CHF Daily Candlesticks — Last 90 Trading Days", y-axis "Price (CHF per USD)". Use mplfinance-style rendering.
- Dependencies: Block 2.1

### Block 2.12: One-Minute Return Distribution
- Type: visualization
- Purpose: Reader sees the empirical distribution of 1-minute log returns — its shape, tails, and whether it approximates a normal distribution.
- Input: df['close']
- Output: Displayed figure, variable returns_1m (Series of log returns)
- Visualization Spec: Histogram with KDE overlay, x = 1-minute log return (pct), y = density, 200 bins, navy histogram bars with 0.5 alpha, orange KDE line, vertical dashed red line at mean (near zero), title "1-Minute Log Return Distribution", x-axis "Log Return", y-axis "Density". Overlay a normal distribution (black dashed) with same mean/std for visual comparison.
- Dependencies: Block 2.1

### Block 2.13: Return Distribution Interpretation
- Type: markdown_narrative
- Purpose: Reader understands the stylized facts of financial returns visible in the histogram: fat tails, excess kurtosis, volatility clustering, and why normality assumptions fail.
- Input: returns_1m from Block 2.12, histogram from Block 2.12
- Output: N/A (inline markdown)
- Narrative Content: Explain: (1) mean ≈ 0 (no drift at 1-min scale), (2) distribution is peaked (leptokurtic) — more density near zero than normal, (3) fat tails — extreme moves happen more often than a normal distribution predicts, (4) these are universal stylized facts of financial returns. Implication: MSE-based models assume normality; this mismatch matters for risk estimation. Note that tick data amplifies microstructure noise at this frequency.
- Dependencies: Block 2.12

---

## Section 3: Preprocessing and Feature Engineering

### Block 3.1: Load Raw OHLC Data
- Type: code_output
- Purpose: Reader learns the raw data structure — 2.3M rows of 1-min USD/CHF with datetime index, 4 price columns, meaningless volume.
- Input: data/processed/USDCHF_1min_2020_2026.csv
- Output: df (DataFrame, datetime-indexed, columns: open, high, low, close), volume dropped, date_range displayed (2020-01-01 → 2026-05-29)
- Dependencies: none

### Block 3.2: Log-Return Target — Definition and Computation
- Type: markdown_narrative
- Purpose: Reader learns why absolute price failed catastrophically (non-stationarity across 6 years), and why log-return ln(close_{t+1} / close_t) solves it via approximate stationarity and scale invariance.
- Narrative Content: Explain v1 failure: MLP R² = -3.26 with absolute price target because price level drifts across regime shifts (2020-21 bullish, 2022-24 sideways, 2025-26 bearish). Log-return removes trend: y_t = ln(close_{t+1} / close_t). Properties: mean ≈ 0, variance roughly constant, dimensionless → model doesn't need to learn absolute scale. Equation: target = log(close.shift(-1) / close). Note: last row gets NaN (no t+1). This makes forecasting a zero-mean prediction problem.
- Dependencies: Block 3.1

### Block 3.3: Compute Log-Return Target
- Type: code_output
- Purpose: Compute and append target column to df — ln(close_{t+1} / close_t).
- Input: df.close
- Output: df['target'] (float64, ~2.3M rows, last row NaN)
- Dependencies: Block 3.1, Block 3.2

### Block 3.4: ADF Stationarity Test on Target
- Type: code_output
- Purpose: Empirically confirm the target is stationary — Augmented Dickey-Fuller test prints test statistic and p-value. p < 0.001 → reject unit root null → target is stationary, validating the log-return transformation.
- Input: df['target'] (non-NaN)
- Output: printed ADF statistic and p-value (e.g. "ADF Statistic: -74.32, p-value: 0.0000")
- Dependencies: Block 3.3

### Block 3.5: Lag Features (8 Features)
- Type: code_output
- Purpose: Create 8 lag features close_lag_k for k ∈ {1,2,3,5,10,15,30,60} — past close prices shifted backward by k minutes. Captures short-term momentum and mean-reversion signals at exponentially increasing horizons.
- Input: df.close
- Output: 8 columns added to df: close_lag_1, close_lag_2, close_lag_3, close_lag_5, close_lag_10, close_lag_15, close_lag_30, close_lag_60 (each with NaN in first k rows)
- Dependencies: Block 3.3

### Block 3.6: Rolling Window Statistics (16 Features)
- Type: code_output
- Purpose: Compute 4 aggregated stats (mean, std, min, max) over 4 windows (5, 10, 30, 60 minutes) on close. Captures local trend, volatility, support/resistance levels at multiple timescales. Uses .rolling(window).agg([...]) with min_periods=1.
- Input: df.close
- Output: 16 columns added to df: close_roll_mean_5, close_roll_std_5, close_roll_min_5, close_roll_max_5, ..._10, ..._30, ..._60
- Dependencies: Block 3.5

### Block 3.7: Price-Derived Features (4 Features)
- Type: code_output
- Purpose: Create 4 interpretable price-ratio features that don't need windows: log_return (instantaneous), pct_change, hl_spread = (high - low)/close (intra-bar volatility), oc_range = (close - open)/close (bar direction and magnitude). All are scale-invariant, immune to absolute price drift.
- Input: df.open, df.high, df.low, df.close
- Output: 4 columns added to df: log_return, pct_change, hl_spread, oc_range
- Dependencies: Block 3.6

### Block 3.8: Technical Indicators (5 Features)
- Type: code_output
- Purpose: Compute 5 classic indicators: rsi_14 (momentum oscillator, 0-100), macd_hist (trend-following, 12-26-9 EMA), bb_position_20 and bb_width_20 (volatility bands, SMA ± 2σ), atr_14 (true range volatility). Provides domain-standard signals traders use.
- Input: df.close, df.high, df.low
- Output: 5 columns added to df: rsi_14, macd_hist, bb_position_20, bb_width_20, atr_14
- Dependencies: Block 3.7

### Block 3.9: Drop NaN Rows — Feature Completeness
- Type: code_output
- Purpose: Remove rows where any feature or target is NaN (from rolling windows, lags, indicators, and last-row target). Reports how many rows remain (~2.0M out of 2.3M). Justification: models require complete feature vectors; alternatives (imputation) would leak future information in time series.
- Input: df (all 34 feature columns + target)
- Output: df (cleaned, ~2.0M rows), printed dropped_count and remaining_count
- Dependencies: Block 3.8

### Block 3.10: Chronological Train/Validation/Test Split
- Type: markdown_narrative
- Purpose: Reader learns why time-series data MUST NOT be shuffled, and how chronological splitting prevents look-ahead bias — models see only past data to predict future, mirroring real deployment.
- Narrative Content: Explain data leakage: if you shuffle, a model can "see" future patterns in training → inflated validation metrics, worthless in production. Chronological split: all timestamps before cutoff go to train, after to val/test. Cutoffs chosen to balance regime coverage: train spans 2020-2024 (bull + sideways), val spans Jan-Sep 2025 (bullish), test spans Sep 2025+ (bearish) — this is hard on purpose, testing generalization across a regime change.
- Dependencies: Block 3.9

### Block 3.11: Execute Chronological Split
- Type: code_output
- Purpose: Split df into X_train, ytr (< 2025-01-01), X_val, yva (2025-01-01 to 2025-09-01), X_test, yte (≥ 2025-09-01). Print row counts: ~1.8M / ~247K / ~276K. Define feature_cols list (34 column names) and separate features from target.
- Input: df (cleaned), cutoffs 2025-01-01, 2025-09-01, feature_cols list
- Output: X_train, ytr, X_val, yva, X_test, yte (all float64, no index leakage to features), feature_cols (list of 34 strings), printed split sizes
- Dependencies: Block 3.9, Block 3.10

### Block 3.12: VIF Multicollinearity Table
- Type: table
- Purpose: Detect multicollinearity among the 34 features. Variance Inflation Factor > 10 flags problematic redundancy; helps reader understand which features carry independent signal. Computed on training set only to avoid data leakage.
- Visualization Spec: Table: column 1 = feature name, column 2 = VIF value. Sorted descending. Color-code: green (VIF < 5), yellow (5-10), red (> 10). Title: "Variance Inflation Factor — Training Set Features"
- Input: X_train (34 features)
- Output: printed VIF table (34 rows × 2 columns)
- Dependencies: Block 3.11

### Block 3.13: Mutual Information Ranking
- Type: visualization
- Purpose: Rank 34 features by mutual information with target — measures nonlinear dependency, complementary to Pearson correlation. Highest MI features are most predictive; low-MI features may be candidates for removal in future iterations.
- Visualization Spec: Horizontal bar chart, x = mutual information score, y = feature names (sorted descending), color = blue gradient (darker = higher MI). Title: "Mutual Information with Log-Return Target (Training Set)"
- Input: X_train, ytr
- Output: printed MI scores + bar chart; mi_scores dict mapping feature name → score
- Dependencies: Block 3.12

### Block 3.14: StandardScaler Normalization
- Type: code_output
- Purpose: Zero-center and unit-variance scale all 34 features. Fit μ and σ on training set ONLY (prevents look-ahead), then transform train, val, test with same parameters. Critical for gradient-based models (MLP) and distance-based models (KNN).
- Input: X_train, X_val, X_test
- Output: X_train_scaled, X_val_scaled, X_test_scaled (all float32, zero-mean unit-variance), scaler_mean_ (34 floats), scaler_scale_ (34 floats)
- Dependencies: Block 3.13

### Block 3.15: Save Preprocessed Tensors to data.pt
- Type: code_output
- Purpose: Persist preprocessed pipeline output as a single .pt file for model scripts to consume without re-running preprocessing (saves 2-3 min per model run). Contains all scaled tensors, feature names, and scaler parameters for inverse-transform during evaluation.
- Input: X_train_scaled, X_val_scaled, X_test_scaled, ytr, yva, yte, feature_cols, scaler_mean_, scaler_scale_
- Output: outputs/preprocessed/data.pt (301MB) containing dict: {'X_train': tensor, 'y_train': tensor, 'X_val': tensor, 'y_val': tensor, 'X_test': tensor, 'y_test': tensor, 'feature_names': list, 'scaler_mean': tensor, 'scaler_scale': tensor}
- Dependencies: Block 3.14

---

## Section 4: Model Training

### Block 4.1: Data Splitting Strategy
- Type: markdown_narrative
- Purpose: Reader understands why a fixed train/val/test split with seed=42 is essential for fair model comparison.
- Input: None (conceptual, referencing the full feature matrix and targets loaded in prior sections)
- Output: None (sets up the split that produces X_train, ytr, X_val, yva, X_test, yte)
- Visualization Spec: None
- Narrative Content: Explain the 3-way split (train/val/test) and why all models must share identical splits. Cover the role of the validation set for hyperparameter tuning and early stopping, the test set held out until final evaluation, and the importance of seed=42 for reproducibility. Mention that the split was already executed in a prior section so the variable names X_train, ytr, X_val, yva, X_test, yte are now available.
- Dependencies: None (references split already done before this section)

### Block 4.2: MLP Architecture Overview
- Type: markdown_narrative
- Purpose: Reader learns the MLP's layer-by-layer architecture before seeing code — width progression, normalization, activation, regularization, and why ~728K parameters is appropriate.
- Input: feature_cols (34 input features from data prep section)
- Output: None (conceptual setup for MLP training blocks)
- Visualization Spec: None
- Narrative Content: Describe the MLP as a deep feedforward network: input layer (34 features) → hidden layers [1024, 512, 256, 128] → output (1). Explain each architectural choice: BatchNorm1d after each linear layer for training stability, ReLU for non-linearity, Dropout(0.15) for regularization. Note total parameter count (~728K) and why this depth/width balances capacity against overfitting on tabular data. Mention the final layer has no activation (raw regression output).
- Dependencies: Block 4.1 (references feature_cols dimension)

### Block 4.3: MLP Training Configuration
- Type: markdown_narrative
- Purpose: Reader understands the optimizer, scheduler, precision, batch size, and early stopping choices — the full training recipe.
- Input: None (conceptual)
- Output: None (sets up MLP training loop semantics)
- Visualization Spec: None
- Narrative Content: Explain each hyperparameter choice: AdamW(lr=0.001, wd=1e-4) — why weight decay decoupled from Adam, why 0.001 is a reasonable starting LR for this architecture. CosineAnnealingWarmRestarts — what warm restarts do to escape local minima, how cosine schedule cycles. AMP (float16) — mixed precision training halves memory and speeds GPU computation with negligible accuracy loss. Batch size 65536 — why massive batches work well on tabular data with GPU, how it stabilizes gradient estimates. Early stopping patience=15 on val loss — what patience means, how the best model checkpoint is saved. 150 max epochs — ceiling if early stop never triggers.
- Dependencies: Block 4.2 (architecture context needed to motivate optimizer/scheduler choices)

### Block 4.4: MLP Training Loop Setup
- Type: code_output
- Purpose: Reader sees the concrete instantiation of model, loss function, optimizer, scheduler, scaler, and dataloaders — all objects needed before the loop runs.
- Input: X_train, ytr, X_val, yva (from Block 4.1 split); feature_cols (input dimension)
- Output: model (MLP instance on GPU), criterion (MSELoss), optimizer (AdamW), scheduler (CosineAnnealingWarmRestarts), scaler (GradScaler for AMP), train_loader and val_loader (DataLoader with batch_size=65536), best_val_loss (initialized to inf), patience_counter (initialized to 0), best_model_state (None, to hold checkpoint)
- Visualization Spec: None
- Dependencies: Block 4.1 (split data), Block 4.2 (architecture), Block 4.3 (hyperparameters)

### Block 4.5: MLP Training Loop Execution
- Type: code_output
- Purpose: Reader observes the per-epoch training dynamics — forward/backward pass with AMP, loss tracking, validation, checkpointing, and early stopping logic.
- Input: model, criterion, optimizer, scheduler, scaler, train_loader, val_loader, best_val_loss, patience_counter, best_model_state (all from Block 4.4)
- Output: train_losses (list of per-epoch training loss), val_losses (list of per-epoch validation loss), best_epoch (int, epoch where best val loss occurred), best_model_state (updated dict with best checkpoint weights), actual_epochs_run (int, number of epochs before early stop)
- Visualization Spec: None (raw data produced; visualization is Block 4.6)
- Dependencies: Block 4.4 (all inputs defined there)

### Block 4.6: MLP Training Curves
- Type: visualization
- Purpose: Reader visually diagnoses training — convergence speed, overfitting onset, and where early stopping fired.
- Input: train_losses, val_losses, best_epoch (from Block 4.5)
- Output: Displayed figure inline in notebook
- Visualization Spec: Dual line plot. x-axis = epoch (1 to actual_epochs_run). y-axis = MSE loss. Blue solid line = training loss (train_losses). Orange solid line = validation loss (val_losses). Vertical dashed red line at x = best_epoch. Title: "MLP Training Curves". Legend in upper right. Grid on. Annotation on red line: "Early stop at epoch {best_epoch}".
- Dependencies: Block 4.5 (loss arrays and best_epoch)

### Block 4.7: KNN Architecture Overview
- Type: markdown_narrative
- Purpose: Reader understands that KNN is a lazy learner — no training phase, prediction is inference-time nearest-neighbor averaging — and why GPU-accelerated distance computation matters at scale.
- Input: None (conceptual)
- Output: None (sets up KNN blocks)
- Visualization Spec: None
- Narrative Content: Contrast KNN with MLP: no weights, no gradient descent, no epochs. "Training" is simply storing the training set. Prediction: for a query point, compute distances to all stored training points, find the k nearest, return the mean of their targets. Explain why brute-force cdist on ~1.8M rows is infeasible per query → subsample 100K train rows for manageable inference. Explain GPU batched cdist (torch.cdist on GPU) for parallel distance computation. Describe the k-value sweep: k ∈ [1,3,5,7,10,15,20,30,50] to study bias-variance tradeoff. Note: val set queries batched at 5000 rows to balance memory.
- Dependencies: Block 4.1 (references X_train, ytr size to motivate subsampling)

### Block 4.8: KNN Subsample and k-Value Sweep
- Type: code_output
- Purpose: Reader sees the random subsample of training data and the loop over k values computing validation predictions and MSE.
- Input: X_train, ytr, X_val, yva (from Block 4.1)
- Output: X_knn_train (100K-row subsample), y_knn_train (corresponding targets), knn_k_list ([1,3,5,7,10,15,20,30,50]), knn_val_results (dict mapping k → val MSE), knn_best_k (int, k with lowest val MSE)
- Visualization Spec: None
- Dependencies: Block 4.1 (train/val data), Block 4.7 (k-value list and methodology)

### Block 4.9: KNN Validation Performance by k
- Type: visualization
- Purpose: Reader visualizes the bias-variance tradeoff — how k impacts validation error and identifies the optimal k.
- Input: knn_k_list, knn_val_results (from Block 4.8)
- Output: Displayed figure inline in notebook
- Visualization Spec: Scatter plot with connecting line. x-axis = k (log scale, values 1,3,5,7,10,15,20,30,50). y-axis = validation MSE. Blue filled circles at each (k, MSE) point, connected by a thin gray line. Vertical dashed green line at x = knn_best_k. Title: "KNN Validation MSE vs. k". Annotation: "Best k = {knn_best_k}" near the green line. Grid on.
- Dependencies: Block 4.8 (k values and MSE results)

### Block 4.10: XGBoost Architecture Overview
- Type: markdown_narrative
- Purpose: Reader understands gradient boosting as sequential tree building, the histogram-based split finding algorithm, and why grid search + CV is the core "training loop" for XGBoost.
- Input: None (conceptual)
- Output: None (sets up XGBoost blocks)
- Visualization Spec: None
- Narrative Content: Explain XGBoost: ensemble of decision trees built sequentially, each tree corrects residuals of prior ensemble. Histogram-based algorithm bins continuous features for efficient split finding — runs on GPU via ROCm (device='cuda', tree_method='hist'). Key hyperparameters: max_depth (tree complexity), learning_rate (shrinkage per tree), subsample (row sampling per tree), colsample_bytree (column sampling per tree), min_child_weight (minimum sum of instance weight in a leaf — regularization). Objective: reg:squarederror for standard MSE regression. Describe the grid search strategy: 216 combinations across 5 hyperparameters, evaluated with 3-fold cross-validation on a 200K-row subsample to make search tractable. Early stopping of 20 rounds within each fold to avoid over-building trees. After best params found, retrain on full 1.8M+ rows. Subsample for search and full training both use GPU (ROCm).
- Dependencies: Block 4.1 (data size context for subsampling rationale)

### Block 4.11: XGBoost Grid Search and Cross-Validation
- Type: code_output
- Purpose: Reader sees the grid definition, the 216-combination Cartesian product, and the 3-fold CV loop with early stopping that finds the best hyperparameter set.
- Input: X_train, ytr (from Block 4.1); param grid definitions
- Output: xgb_param_grid (dict: max_depth=[5,7,9,11], learning_rate=[0.01,0.03,0.05], subsample=[0.7,0.8,0.9], colsample_bytree=[0.6,0.8], min_child_weight=[1,3,5]), xgb_search_results (dataframe: 216 rows, columns = params + mean_cv_score + std_cv_score), xgb_best_params (dict of best hyperparameter combination), xgb_best_cv_score (float, best mean validation MSE from 3-fold CV)
- Visualization Spec: None
- Dependencies: Block 4.1 (train data), Block 4.10 (hyperparameter semantics)

### Block 4.12: XGBoost Full Training with Best Parameters
- Type: code_output
- Purpose: Reader sees the final model trained on the entire training set with the best hyperparameters — the deliverable trained XGBoost object.
- Input: X_train, ytr (all 1.8M+ rows), X_val, yva, xgb_best_params (from Block 4.11)
- Output: xgb_model (trained XGBRegressor object), xgb_train_losses (per-iteration training loss from evals_result), xgb_val_losses (per-iteration validation loss from evals_result), xgb_best_iteration (int, iteration where early stopping fired), xgb_actual_boost_rounds (total trees built)
- Visualization Spec: None
- Dependencies: Block 4.11 (best params), Block 4.1 (full data)

### Block 4.13: XGBoost Training Curves
- Type: visualization
- Purpose: Reader visually inspects XGBoost training — convergence of boosting rounds, validation loss trajectory, and where early stopping triggered.
- Input: xgb_train_losses, xgb_val_losses, xgb_best_iteration (from Block 4.12)
- Output: Displayed figure inline in notebook
- Visualization Spec: Dual line plot. x-axis = boosting round (1 to xgb_actual_boost_rounds). y-axis = RMSE or MSE loss. Blue solid line = training loss. Orange solid line = validation loss. Vertical dashed red line at x = xgb_best_iteration. Title: "XGBoost Training Curves". Legend upper right. Grid on. Annotation: "Best iteration: {xgb_best_iteration}".
- Dependencies: Block 4.12 (loss arrays and best iteration)

### Block 4.14: Model Training Summary
- Type: markdown_narrative
- Purpose: Reader synthesizes the three training approaches — their fundamentally different philosophies (gradient descent vs. lazy vs. boosting), computational requirements (GPU vs. GPU-distance vs. GPU-hist), and training dynamics.
- Input: best_epoch (Block 4.5), knn_best_k and knn_val_results (Block 4.8), xgb_best_iteration and xgb_best_params (Block 4.11/4.12)
- Output: None (narrative synthesis)
- Visualization Spec: None
- Narrative Content: Compare the three approaches: MLP learns a continuous nonlinear function via SGD — ~700K parameters, GPU, 150-epoch ceiling, early stopping after convergence. KNN stores data and queries at inference time — zero trainable parameters, GPU for distance math, k controls smoothness. XGBoost greedily adds trees — sequential, GPU histogram (ROCm), best hyperparams found via 216-combo grid search + 3-fold CV. Contrast training times, memory footprints, and the cost of hyperparameter search. Preview that next section (Section 5) will evaluate all three on the held-out test set using evals and preds_dict.
- Dependencies: Block 4.5, Block 4.8, Block 4.9, Block 4.11, Block 4.12

---

## Section 5: Evaluation

### Block 5.1: Section Header — "5. Evaluation"
- Type: section_header
- Purpose: Frame the evaluation as a multi-model, multi-angle assessment — metrics alone are insufficient; residuals, direction, clustering, and PCA reveal deeper model behavior.
- Input: (none — structural marker)
- Output: Rendered section heading in notebook
- Dependencies: (none)

### Block 5.2: Compute Regression Metrics
- Type: code_output
- Purpose: Compute the five core regression metrics (RMSE, MAE, R², MAPE, Direction Accuracy) for all three models to populate the comparison table.
- Input: yte (true test labels), preds_dict (dict of model_name → y_pred arrays for MLP, KNN, XGBoost)
- Output: metrics_df — pandas DataFrame with index = model names, columns = ['RMSE', 'MAE', 'R²', 'MAPE (%)', 'Direction Accuracy (%)']
- Dependencies: Block 5.1

### Block 5.3: Regression Metrics Table
- Type: table
- Purpose: Display the numeric regression metrics in a formatted table so the reader can compare all three models at a glance.
- Input: metrics_df (from Block 5.2)
- Output: Rendered table in notebook (best value per column highlighted in bold)
- Dependencies: Block 5.2

### Block 5.4: Residual Distribution — Histogram + KDE
- Type: visualization
- Purpose: Reveal the shape, center, and spread of residual distributions per model — zero-centered and symmetric residuals indicate unbiased predictions.
- Input: yte, preds_dict
- Output: residual_dist.png — 1×3 subplot grid
- Visualization Spec: 1 row × 3 columns. Each subplot: histogram of residuals (residual = y_true − y_pred) for one model, bins=50, with overlaid KDE curve. X-axis: residual value (shared scale across subplots). Y-axis: density. Color: consistent per-model color (MLP=#2ca02c green, KNN=#ff7f0e orange, XGBoost=#1f77b4 blue). Title per subplot: model name + skewness/kurtosis annotation. Vertical dashed line at x=0.
- Dependencies: Block 5.2

### Block 5.5: Residual QQ Plot — Normality Check
- Type: visualization
- Purpose: Assess whether residuals follow a Gaussian distribution — deviations from the diagonal reveal heavy tails or skew, which violate standard regression assumptions.
- Input: yte, preds_dict
- Output: qq_plot.png — 1×3 subplot grid
- Visualization Spec: 1 row × 3 columns. Each subplot: scipy.stats.probplot QQ plot (theoretical quantiles vs sample quantiles) for one model's residuals. X-axis: theoretical normal quantiles. Y-axis: ordered residual values. Red diagonal reference line (y=x scaled to data). Model colors same as Block 5.4. Title: model name. If residuals are normal, points lie on the line.
- Dependencies: Block 5.4

### Block 5.6: Residual ACF Plot — Time-Series Independence
- Type: visualization
- Purpose: Test whether residuals exhibit serial correlation — significant autocorrelation at any lag means the model fails to capture temporal structure, undermining i.i.d. assumptions.
- Input: yte, preds_dict
- Output: acf_plot.png — 1×3 subplot grid
- Visualization Spec: 1 row × 3 columns. Each subplot: statsmodels ACF correlogram of residuals for one model, lags 0–40. Blue bars = autocorrelation values. Light blue shaded region = 95% confidence interval (±1.96/√n). X-axis: lag. Y-axis: autocorrelation (−1 to 1). Title: model name. Bars crossing the shaded band are statistically significant.
- Dependencies: Block 5.4

### Block 5.7: Actual vs Predicted Scatter
- Type: visualization
- Purpose: Visualize prediction quality — points clustering tightly along the diagonal indicate strong fit; systematic deviations (fanning, curvature, offset) expose model weaknesses.
- Input: yte, preds_dict
- Output: scatter_actual_vs_pred.png — single figure with 3 overlaid scatter series
- Visualization Spec: Single axes. Three scatter series overlaid: XGBoost (blue, alpha=0.3), KNN (orange, alpha=0.3), MLP (green, alpha=0.3). X-axis: y_true (actual). Y-axis: y_pred (predicted). Black dashed diagonal y=x reference line (perfect prediction). Legend identifies each model. Annotations: R² value per model in matching color in corner or legend.
- Dependencies: Block 5.2

### Block 5.8: Direction Classification — Compute
- Type: code_output
- Purpose: Convert regression outputs to binary direction labels (up/down) and compute confusion matrices plus precision/recall/F1 metrics — this evaluates whether models correctly predict the sign of movement, independent of magnitude accuracy.
- Input: yte, preds_dict
- Output: dir_cm_dict — dict of model_name → 2×2 confusion matrix (rows=true, cols=pred; [0,0]=TN down-correct, [0,1]=FP down-wrong, [1,0]=FN up-wrong, [1,1]=TP up-correct). dir_metrics_df — DataFrame with index = model names, columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score'].
- Computation: For each model, compute y_true_dir = (yte > 0).astype(int), y_pred_dir = (y_pred > 0).astype(int). Then sklearn confusion_matrix and classification_report → extracted metrics.
- Dependencies: Block 5.2

### Block 5.9: Direction Classification — Confusion Matrix Heatmaps
- Type: visualization
- Purpose: Visualize per-model confusion matrices as annotated heatmaps so the reader can see where direction errors occur (false up-calls vs false down-calls).
- Input: dir_cm_dict (from Block 5.8)
- Output: dir_confusion_matrix.png — 1×3 subplot grid
- Visualization Spec: 1 row × 3 columns. Each subplot: seaborn heatmap of 2×2 confusion matrix. Cells annotated with count + row-normalized percentage. Colormap: Blues (darker = higher count). X-axis labels: Predicted Down, Predicted Up. Y-axis labels: Actual Down, Actual Up. Title: model name. Consistent model colors in title text.
- Dependencies: Block 5.8

### Block 5.10: Direction Classification — Metrics Table
- Type: table
- Purpose: Display direction classification metrics (Accuracy, Precision, Recall, F1) for all three models side by side.
- Input: dir_metrics_df (from Block 5.8)
- Output: Rendered table in notebook (best value per column bolded)
- Dependencies: Block 5.8

### Block 5.11: Direction Classification — Narrative
- Type: markdown_narrative
- Purpose: Explain the critical distinction between regression accuracy and directional accuracy — a model can have low RMSE but poor direction guessing, or vice versa. Clarify that this binary classification is a derived view, not a separate model.
- Input: dir_metrics_df (from Block 5.8)
- Narrative Content:
  - Define the conversion: y_true > 0 → class 1 (up), y_pred > 0 → class 1 (up). Emphasize: this is DIRECTION only, not return-magnitude classification.
  - Explain each metric in this context: Accuracy = % of directions correctly called. Precision = when model says "up," how often is it right? Recall = of all actual up-moves, how many did model catch? F1 = harmonic mean, balances precision/recall.
  - Note: baseline accuracy is ~50% for balanced up/down splits; models should exceed this.
  - Compare direction accuracy to the Direction Accuracy (%) column from Block 5.3 (they should match — note this consistency check).
  - Highlight if any model has high RMSE but strong direction accuracy (or vice versa) — tradeoff interpretation.
- Dependencies: Block 5.8, Block 5.10

### Block 5.12: Model Comparison — Grouped Bar Charts
- Type: visualization
- Purpose: Enable direct visual comparison of all five regression metrics and training time across the three models in a single figure.
- Input: metrics_df (from Block 5.2), evals dict (for training_time per model)
- Output: model_comparison.png — 2×3 subplot grid
- Visualization Spec: Top row (5 subplots): grouped bar charts, one per metric (RMSE, MAE, R², MAPE, Direction Accuracy). Each subplot: 3 bars side by side (XGBoost blue, KNN orange, MLP green), x-axis = model name, y-axis = metric value. Metric name as subplot title. Bottom row, 1 subplot spanning full width: grouped bar chart for training time (seconds), same color scheme. Value annotations on top of each bar. Consistent y-axis lower bound at 0 for MAE/RMSE/MAPE/time; R² may go negative (set y_min = min(0, min_r2 - 0.1)).
- Dependencies: Block 5.2, Block 5.3

### Block 5.13: Silhouette Analysis — Compute + Plot
- Type: visualization
- Purpose: Determine the natural number of clusters in the feature space by evaluating KMeans silhouette scores for k=2..8 — higher silhouette = better-defined, more separable clusters.
- Input: X_test (20K random subsample from test set), selected_features = ['log_return', 'hl_spread', 'rsi_14', 'macd_hist', 'bb_position_20', 'atr_14']
- Output: silhouette_plot.png — single line plot. best_k (integer), silhouette_scores (list of floats per k)
- Visualization Spec: Line plot with markers. X-axis: k (number of clusters, 2 through 8). Y-axis: silhouette score (range −1 to 1). Blue line with circular markers at each k. Annotate the best k with a star marker and text label ("Best k = X, score = Y.YYY"). Horizontal reference line at y=0 (worse than random). Y-axis lower bound = 0 or min(0, min_score − 0.05). Title: "Silhouette Score vs Number of Clusters". Compute: sklearn KMeans(n_clusters=k, random_state=42).fit_predict() + silhouette_score() per k.
- Dependencies: Block 5.1

### Block 5.14: Silhouette Interpretation — Narrative
- Type: markdown_narrative
- Purpose: Explain what silhouette scores mean in this financial feature context and what the best k reveals about the data structure.
- Input: best_k, silhouette_scores (from Block 5.13)
- Narrative Content:
  - Define silhouette: per-sample measure = (b − a) / max(a, b), where a = mean intra-cluster distance, b = mean nearest-cluster distance. Range [−1, 1]. High positive = sample well-matched to own cluster, far from others. Near 0 = boundary sample. Negative = likely misclassified.
  - Cluster-average silhouette is the mean across all samples. Used for k selection.
  - Interpret the observed curve: if scores are low overall (<0.3), the feature space lacks strong cluster structure — data is fairly continuous. If scores peak at small k and decline, natural clusters exist but are few. If scores increase with k, the data resists partitioning.
  - Connect to domain: what does it mean for financial features like RSI, MACD, and log-returns to have (or lack) distinct clusters? Market regimes? Volatility states?
  - State the chosen best k and its silhouette score, and what practical use KMeans labels serve downstream (e.g., coloring PCA plots in Block 5.15).
- Dependencies: Block 5.13

### Block 5.15: PCA — 2-Component Projection
- Type: visualization
- Purpose: Project the high-dimensional test feature space into 2D to visually assess structure, cluster coherence, and error distribution patterns.
- Input: X_test (30K random subsample), feature_cols (all feature columns), best_k + KMeans labels (from Block 5.13), preds_dict (for XGBoost errors), yte
- Output: pca_views.png — 1×3 subplot grid
- Visualization Spec: All subplots share the same 2D PCA projection (PC1 on x-axis, PC2 on y-axis). PCA computed via sklearn PCA(n_components=2) on standardized 30K samples. Variance explained annotated per axis label (e.g., "PC1 (34.2%)").
  - Subplot 1 — Colored by KMeans Cluster: Scatter plot, each point colored by its KMeans cluster label (from Block 5.13 best_k). Discrete colormap (tab10). Title: "PCA Colored by KMeans Cluster (k={best_k})". Reveals whether clusters form coherent, separable groups in PCA space.
  - Subplot 2 — Colored by |log_return|: Scatter plot, each point colored by absolute log-return magnitude (continuous colormap: viridis or RdYlGn). Colorbar labeled "|log_return|". Title: "PCA Colored by |log-return|". Reveals whether extreme returns cluster in specific PCA regions.
  - Subplot 3 — Colored by XGBoost |error|: Scatter plot, each point colored by |yte − XGBoost_pred|. Continuous colormap: hot/inferno. Colorbar labeled "|XGBoost Error|". Title: "PCA Colored by XGBoost |Error|". Reveals whether prediction errors concentrate in specific regions of feature space (systematic failure zones).
  - Point size: small (s=1 or s=2), alpha=0.4–0.6 for density readability with 30K points.
- Dependencies: Block 5.13

### Block 5.16: Learning Curves — MLP Train/Val Loss
- Type: visualization
- Purpose: Diagnose MLP convergence and overfitting by plotting training and validation loss per epoch — diverging curves signal overfitting; flat validation loss signals saturation.
- Input: MLP training history object (mlp_history or extracted from evals['MLP']['loss_curve']) — containing train_loss and val_loss arrays per epoch
- Output: learning_curves.png — single dual-line plot
- Visualization Spec: Dual line plot. X-axis: epoch (1 to N, where N = total training epochs). Y-axis: loss (MSE or log-scale depending on range). Blue solid line = training loss. Orange solid line = validation loss. Legend: "Training Loss", "Validation Loss". Vertical dashed red line at early stopping epoch (if applicable; annotate "Early Stop"). If validation loss minimum occurs at epoch M, annotate with marker and text: "Best Val Loss: X.XXXX (epoch M)". Title: "MLP Learning Curves — Training vs Validation Loss". Grid lines on. If overfitting is visible (val loss rising while train loss drops), add annotation: "Overfitting begins ~epoch K".
- Dependencies: Block 5.1

---

## Section 6: Experimental Scenarios

### Block 6.1: Scenario 1 — Parameter Tuning Introduction & Hypothesis
- Type: markdown_narrative
- Purpose: Frames the experiment: optimal hyperparameters improve generalization; KNN sensitive to k/weighting, XGBoost sensitive to learning rate/depth.
- Input: None (section opening)
- Output: None (displayed narrative)
- Narrative Content: State hypothesis: "KNN performance will peak at moderate k (~10–50) with distance weighting outperforming uniform; XGBoost will show a sweet spot at moderate learning rate (~0.05–0.1) and depth (~5–7)." Define RMSE as evaluation metric. Explain why grid search on subsampled data is used (computational budget). Note: all models use feature_cols from prior preprocessing.
- Dependencies: None (first block in section)

### Block 6.2: KNN Hyperparameter Grid Search Sweep
- Type: code_output
- Purpose: Execute 2D grid search over k × weights on 100K train subsample, evaluate on X_val/yva, producing raw results for table and plot.
- Input: X_train, ytr, X_val, yva, feature_cols
- Output: knn_grid_df (DataFrame: columns k, weight, rmse_val); knn_best_k, knn_best_weight
- Dependencies: Block 6.1

### Block 6.3: KNN Parameter Tuning Results Table
- Type: table
- Purpose: Display all (k, weight) combinations with RMSE so reader can compare across the full grid.
- Input: knn_grid_df
- Output: Rendered table: columns = [k, Weight, RMSE]. Best row highlighted (bold or colored).
- Visualization Spec: Not a chart — formatted table. 16 rows (8 k values × 2 weight schemes). Best row highlighted with background color.
- Dependencies: Block 6.2

### Block 6.4: KNN RMSE vs k Dual-Line Plot
- Type: visualization
- Purpose: Visualize performance curve — RMSE as function of k under both weighting schemes, highlighting optimal k.
- Input: knn_grid_df
- Output: Displayed figure (suggested filename: fig_knn_tuning.png)
- Visualization Spec: Dual line plot. x-axis = k (logarithmic scale, values [1,3,5,10,20,50,100,200]). y-axis = RMSE. Blue solid line with circle markers = uniform weight. Orange dashed line with triangle markers = distance weight. Vertical dashed line (red) at best k. Annotation box: "Best: k=X, weight=Y, RMSE=Z". Legend top-right. Title: "KNN Hyperparameter Tuning — RMSE vs k"
- Dependencies: Block 6.2

### Block 6.5: XGBoost Hyperparameter Grid Search Sweep
- Type: code_output
- Purpose: Execute 2D grid search over learning_rate × max_depth on 200K train subsample with 200 trees, producing raw results for heatmap.
- Input: X_train, ytr, X_val, yva, feature_cols
- Output: xgb_grid_df (DataFrame: columns lr, max_depth, rmse_val); xgb_best_lr, xgb_best_depth, xgb_best_rmse
- Dependencies: Block 6.1

### Block 6.6: XGBoost Learning Rate vs Depth RMSE Heatmap
- Type: visualization
- Purpose: Show interaction surface of two hyperparameters — where combinations produce low RMSE.
- Input: xgb_grid_df
- Output: Displayed figure (suggested filename: fig_xgb_tuning_heatmap.png)
- Visualization Spec: Heatmap. x-axis = learning_rate ([0.001, 0.01, 0.05, 0.1, 0.3]). y-axis = max_depth ([2, 3, 5, 7, 10]). Color = RMSE (coolwarm colormap: blue=low RMSE, red=high RMSE). Numeric RMSE value annotated in each cell. Best cell outlined with thick gold border. Colorbar labeled "RMSE". Title: "XGBoost Hyperparameter Tuning — RMSE Heatmap"
- Dependencies: Block 6.5

### Block 6.7: Scenario 2 — Feature Ablation Introduction & Hypothesis
- Type: markdown_narrative
- Purpose: Frames the experiment: measuring how many features each model needs to perform well; tests robustness to dimensionality reduction.
- Input: None (new scenario)
- Output: None (displayed narrative)
- Narrative Content: State hypothesis: "MLP and XGBoost will retain performance with top 15 features but degrade with 5; KNN will degrade more sharply due to curse of dimensionality in reverse (too few features lose signal)." Define mutual information (MI) as feature importance metric: I(X_i; y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]. Explain procedure: compute MI on X_train/ytr, rank features, select top N, retrain each model on reduced feature set, evaluate on X_test/yte.
- Dependencies: None (new scenario start)

### Block 6.8: Mutual Information Feature Importance Computation
- Type: code_output
- Purpose: Compute MI scores for all 34 features, rank them, and select top-15 and top-5 feature subsets used by all three models.
- Input: X_train, ytr, feature_cols
- Output: mi_scores (Series, len=34, indexed by feature name, sorted descending); top15_features (list of 15 feature names); top5_features (list of 5 feature names)
- Dependencies: Block 6.7

### Block 6.9: Feature Ablation Results Table
- Type: table
- Purpose: Compare RMSE, MAE, R², and train time across all 3 models under 3 feature budgets.
- Input: evals dict with keys: mlp_34, mlp_15, mlp_5, knn_34, knn_15, knn_5, xgb_34, xgb_15, xgb_5 — each containing rmse, mae, r2, train_time.
- Output: Rendered table: columns = [Model, Feature Count, RMSE, MAE, R², Train Time (s)]. 9 rows. Sorted by Model then Feature Count descending. Best RMSE per model highlighted.
- Visualization Spec: Formatted table. Model column merged/grouped (MLP rows together, KNN rows together, XGBoost rows together). Bold best RMSE value per model group.
- Dependencies: Block 6.8

### Block 6.10: RMSE vs Feature Count 3-Line Plot
- Type: visualization
- Purpose: Visual comparison of how each model's error grows as features are removed — reveals which model is most feature-dependent.
- Input: Feature ablation results (extracted from evals into ablation_plot_df with columns: model, n_features, rmse)
- Output: Displayed figure (suggested filename: fig_feature_ablation.png)
- Visualization Spec: Line chart. x-axis = Feature Count ([5, 15, 34], linear scale). y-axis = RMSE. Three color-coded lines with markers: blue = MLP, orange = KNN, green = XGBoost. Marker at each point. Solid lines connecting. Legend upper-right. Title: "Feature Ablation — RMSE vs Number of Features"
- Dependencies: Block 6.9

### Block 6.11: Scenario 3 — Train Size Sensitivity Introduction & Hypothesis
- Type: markdown_narrative
- Purpose: Frames the experiment: how much training data is enough; quantifies diminishing returns of adding more samples.
- Input: None (new scenario)
- Output: None (displayed narrative)
- Narrative Content: State hypothesis: "RMSE will drop steeply from 10% to 50% and plateau beyond 75%, indicating that ~75% of the training data captures most learnable signal." Define learning curve concept. Explain procedure: fixed seed random subsampling at 5 fractions, train XGBoost with best parameters from Scenario 1 (xgb_best_lr, xgb_best_depth), evaluate on full X_test/yte. Note why XGBoost is chosen (fastest to train, best performer in prior experiments).
- Dependencies: Block 6.5 (needs xgb_best_lr, xgb_best_depth)

### Block 6.12: Train Size Sensitivity Sweep
- Type: code_output
- Purpose: Train XGBoost on 5 progressively larger random subsamples of training data with fixed seed, evaluate on full test set.
- Input: X_train, ytr, X_test, yte, feature_cols, xgb_best_lr, xgb_best_depth
- Output: size_results_df (DataFrame: columns frac, n_train_samples, rmse_test, r2_test, train_time_s). 5 rows for [0.10, 0.25, 0.50, 0.75, 1.00].
- Dependencies: Block 6.11, Block 6.5

### Block 6.13: Train Size Sensitivity Results Table
- Type: table
- Purpose: Tabulate RMSE, R², and training time at each fraction so reader sees exact tradeoff.
- Input: size_results_df
- Output: Rendered table: columns = [Train %, Train Samples, RMSE, R², Train Time (s)]. Train % formatted as "10%", "25%", etc. Best RMSE row highlighted.
- Visualization Spec: Formatted table. 5 data rows. Full 100% row bolded as reference baseline.
- Dependencies: Block 6.12

### Block 6.14: Learning Curve — RMSE vs Train Samples
- Type: visualization
- Purpose: Show classic learning curve shape — steep initial improvement, diminishing returns, plateau.
- Input: size_results_df
- Output: Displayed figure (suggested filename: fig_learning_curve.png)
- Visualization Spec: Single line plot with markers. x-axis = Train Samples (logarithmic scale). y-axis = RMSE. Blue line with circle markers. Shaded 95% confidence band (if bootstrap replicates available; otherwise omit). Horizontal dashed gray line at final RMSE (100% train) labeled "asymptotic RMSE = X". Title: "XGBoost Learning Curve — RMSE vs Training Set Size". Annotation: "Diminishing returns beyond ~75%"
- Dependencies: Block 6.12

### Block 6.15: Cross-Scenario Summary Table
- Type: table
- Purpose: One-glance summary: key finding, best config, and RMSE from each scenario for reader synthesis.
- Input: knn_best_k, knn_best_weight (Block 6.2), knn_grid_df (Block 6.2), xgb_best_lr, xgb_best_depth, xgb_grid_df (Block 6.5), ablation_plot_df (Block 6.10), size_results_df (Block 6.12)
- Output: Rendered table: columns = [Scenario, Key Finding, Best Config, Best RMSE]. 3 rows:
  - Row 1: "1. Parameter Tuning" | "Distance weighting and moderate k/depth critical" | "KNN: k=best_k, weight=best_weight; XGB: lr=best_lr, depth=best_depth" | best RMSE from each
  - Row 2: "2. Feature Ablation" | "15 features sufficient; <5 features collapse performance" | "Top 15 features, XGBoost" | best RMSE from ablation
  - Row 3: "3. Train Size Sensitivity" | "~75% data captures nearly all signal" | "75% subsample" | RMSE at 75%
- Visualization Spec: Formatted summary table. Bold key numbers. Scenario column as row headers.
- Dependencies: Block 6.2, Block 6.5, Block 6.9, Block 6.12

---

## Section 7: Conclusion

### Block 7.1: Conclusion Section Header
- Type: section_header
- Purpose: Delimit the start of the conclusion and signal synthesis mode to the reader.
- Input: None
- Output: Rendered heading "7. Conclusion"
- Dependencies: None

### Block 7.2: Key Findings — Model Performance Summary
- Type: markdown_narrative
- Purpose: Reader learns which model won (XGBoost) and why — it was the only model to achieve positive R² (+0.006) on the test set, extracting a tiny but real signal from noisy 1-min forex data.
- Input: evals (dict with keys mlp, knn, xgb, each containing rmse, r2, mape, dir_acc)
- Output: Narrative text covering: (a) XGBoost achieves RMSE=0.000130, R²=+0.006, DirAcc=46.6% — best on all metrics; (b) KNN close second (RMSE=0.000132, R²=-0.015, DirAcc=46.2%); (c) MLP struggles (RMSE=0.000151, R²=-0.331, DirAcc=45.8%) despite largest capacity — deep networks overfit noise when signal is extremely weak; (d) all MAPE < 0.01% means price predictions are highly precise even if direction is near-random.
- Dependencies: evals dict (produced in Section 5)

### Block 7.3: Final Comparison Table
- Type: table
- Purpose: Reader sees all models side-by-side across all metrics for at-a-glance comparison.
- Input: evals (per-model metrics), model training times from training cells
- Output: Markdown table with columns: Model | RMSE ↓ | MAE | R² ↑ | MAPE% ↓ | DirAcc% ↑ | Train Time. Rows: MLP (GPU), KNN (GPU), XGBoost (ROCm GPU). Best value in each column bolded.
- Dependencies: evals (Section 5), training times from Section 4

### Block 7.4: Direction Accuracy Ceiling — Why ~47% Is Expected
- Type: markdown_narrative
- Purpose: Reader understands that ~46-47% directional accuracy does not mean the model is useless — it reflects the efficient-market / random-walk nature of 1-min forex where the theoretical ceiling for directional prediction on a single instrument without external information is just above 50%.
- Input: evals.*.dir_acc, confusion matrix metrics from Section 5 (cm_mlp, cm_knn, cm_xgb with precision/recall/F1)
- Output: Narrative explaining: (a) efficient market hypothesis implies near-50% directional accuracy ceiling for any model using only price history; (b) 46.6% vs 50% baseline means the model captures a real but tiny edge; (c) regression metrics (RMSE/R²) are the appropriate evaluation framework, not classification; (d) precision/recall breakdown shows up-moves are slightly more predictable than down-moves (if true from confusion matrix data).
- Dependencies: evals (Section 5), confusion matrix results (Section 5)

### Block 7.5: XGBoost Feature Importance — Code Output
- Type: code_output
- Purpose: Extract which of the 34 engineered features most influence XGBoost predictions, computed from the trained model's gain-based importance scores.
- Input: model_xgb (trained XGBoost booster from Section 4), feature_cols (list of 34 feature names from Section 3)
- Output: feature_importance (DataFrame with columns: feature, importance_gain, importance_pct; sorted descending by gain; top 15 rows)
- Dependencies: model_xgb (Section 4), feature_cols (Section 3)

### Block 7.6: XGBoost Feature Importance — Visualization
- Type: visualization
- Purpose: Reader sees which engineered features dominate XGBoost decisions and understands that price-derived features (log_return, hl_spread) and recent lags matter more than technical indicators.
- Input: feature_importance (from Block 7.5)
- Output: Horizontal bar chart. x-axis: importance score (gain). y-axis: feature name. Top 15 features only. Color: single blue gradient (darker = higher importance). Title: "XGBoost Feature Importance — Top 15 Features". Annotation: total explained gain % in top-right corner.
- Visualization Spec: Horizontal bar chart, x=importance_gain, y=feature (top 15 descending), color=#2563eb gradient, title="XGBoost Feature Importance — Top 15 Features", annotation at top-right showing cumulative importance percentage.
- Dependencies: Block 7.5

### Block 7.7: Effect of Training Set Size — Code Output
- Type: code_output
- Purpose: Quantify how XGBoost performance scales with training data volume by retraining at 1%, 5%, 10%, 25%, 50%, 100% fractions of the full train set and evaluating on the validation set.
- Input: X_train, ytr, X_val, yva, best XGBoost params from Section 4 (best_params_xgb: max_depth=5, lr=0.05, subsample=0.9, colsample_bytree=0.6, min_child_weight=5, n_estimators=65)
- Output: train_size_results (dict: fraction → {rmse, r2, dir_acc, train_time_sec})
- Dependencies: X_train, ytr, X_val, yva (Section 3), best_params_xgb (Section 4)

### Block 7.8: Effect of Training Set Size — Visualization
- Type: visualization
- Purpose: Reader sees that XGBoost benefits substantially from more data up to ~50% of the training set, then shows diminishing returns — confirming that 1.8M samples is sufficient.
- Input: train_size_results (from Block 7.7)
- Output: Dual-axis line plot. x-axis: training set fraction (1%, 5%, 10%, 25%, 50%, 100% on log scale). Left y-axis: RMSE (blue line, circles). Right y-axis: Direction Accuracy % (orange line, triangles). Title: "Scaling Behavior: XGBoost Performance vs Training Set Size". Horizontal dashed line at RMSE=0.000130 (full-size baseline). Annotation: "Diminishing returns beyond ~50%" with arrow at 50% point.
- Visualization Spec: Dual-axis line plot, x=fraction (log scale, labels: 1%, 5%, 10%, 25%, 50%, 100%), y1=RMSE (blue #2563eb line with circle markers), y2=Direction Accuracy % (orange #f59e0b line with triangle markers), horizontal dashed line at y1=0.000130 labeled "Full train set baseline", annotation arrow at x=50% with text "Diminishing returns", title="Scaling Behavior: XGBoost Performance vs Training Set Size"
- Dependencies: Block 7.7

### Block 7.9: Model Recommendation for Deployment
- Type: markdown_narrative
- Purpose: Reader learns that XGBoost is the recommended model for deployment, with explicit trade-off analysis: it has the best accuracy but longest training time (14.8m grid search; 5s inference), while MLP offers fastest inference (GPU batch) at the cost of negative R², and KNN offers middle ground but requires storing the full 100K-sample reference set.
- Input: evals (all metrics), model file sizes (mlp_v2.pt = 2.8MB, xgboost_v2.json = 281KB, KNN reference set size), training/inference times
- Output: Narrative covering: (a) XGBoost recommended — best R², best DirAcc, smallest model file (281KB), fast inference (5s on 276K test samples); (b) MLP trade-off — fast GPU inference but unreliable predictions (R² negative, meaning it's worse than predicting the mean); (c) KNN trade-off — competitive accuracy but requires storing 100K reference samples in memory, inference time scales with test set size; (d) for production, retrain XGBoost weekly on latest data with same hyperparams.
- Dependencies: evals (Section 5), model file metadata

### Block 7.10: Study Limitations
- Type: markdown_narrative
- Purpose: Reader understands the boundary conditions of this study so they don't overgeneralize: 1-min data is inherently noisy (bid-ask bounce, microstructure noise), no macro-economic features (interest rates, news sentiment, correlated pairs), single currency pair (USD/CHF may not generalize to other pairs), and no transaction cost modeling (spread, slippage, commission would erode the tiny edge).
- Input: None (describes study design constraints)
- Output: Narrative listing each limitation with its practical implication: (a) 1-min noise → signal-to-noise ratio is extremely low, limiting ceiling; (b) no external features → model only sees price history, missing macro drivers; (c) single pair → results may not transfer to other currency pairs with different volatility regimes; (d) no transaction costs → profitable deployment would require spread < 0.013% (RMSE scale) which is unrealistic for retail; (e) chronological split ensures temporal validity but single split means metrics have variance.
- Dependencies: None (but positioned after deployment block for logical flow)

### Block 7.11: Future Work
- Type: markdown_narrative
- Purpose: Reader learns the most promising next research directions: ensemble methods to combine model strengths, attention-based architectures (LSTM/Transformer) for temporal patterns, multi-currency modeling for cross-pair signal, longer forecast horizons (5min, 15min, 1hr), Optuna hyperparameter optimization, and time-series cross-validation.
- Input: Future improvement ideas from project plan
- Output: Narrative listing 6 future directions, each with a 1-sentence rationale: (a) Ensemble (stack XGBoost + KNN) — may push DirAcc past 47% by combining tree and distance-based signals; (b) LSTM/Transformer — sequence models can capture multi-step temporal dependencies that feedforward architectures miss; (c) Multi-currency — correlated pairs (EUR/USD, USD/JPY) provide additional predictive signal; (d) Longer horizons — 5-min/15-min/1-hour predictions may have higher signal-to-noise ratio than 1-min; (e) Optuna hyperparameter optimization — smarter search than manual grid may find better XGBoost/MLP configs; (f) TimeSeriesSplit cross-validation — expanding-window CV would quantify metric variance across time periods and detect regime-dependent performance.
- Dependencies: None

### Block 7.12: Final Takeaways
- Type: markdown_narrative
- Purpose: Reader leaves with 3 concise, memorable takeaways that synthesize the entire study into actionable conclusions.
- Input: All previous blocks (synthesizes evals, feature_importance, train_size_results, limitation analysis)
- Output: Three bullet-point takeaways: (1) "XGBoost with 34 engineered features achieves the only positive R² (+0.006) on 1-min USD/CHF — tree-based methods handle noisy, non-linear forex data better than neural networks or k-NN." (2) "Directional accuracy ceiling of ~46-47% is inherent to the problem — 1-min forex is near-random-walk, and the model's edge comes from magnitude prediction (RMSE 0.000130) rather than directional bets." (3) "Log-return transformation and chronological splitting are non-negotiable for multi-year time series — they prevent regime-shift failure (v1 R² was -3.26)."
- Dependencies: Blocks 7.2–7.11

---

## Cell Count Estimate

| Section | Title | Blocks | Markdown | Code/Output | Table | Visualization | Total Cells |
|---------|-------|--------|----------|-------------|-------|---------------|-------------|
| 1 | Project Description | 11 | 10 | 0 | 0 | 0 | 11 |
| 2 | Exploratory Data Analysis | 13 | 4 | 1 | 3 | 4 | 13 |
| 3 | Preprocessing & Feature Engineering | 15 | 2 | 10 | 1 | 1 | 15 |
| 4 | Model Training | 14 | 5 | 5 | 0 | 3 | 14 |
| 5 | Evaluation | 16 | 3 | 3 | 3 | 7 | 16 |
| 6 | Experimental Scenarios | 15 | 3 | 4 | 4 | 4 | 15 |
| 7 | Conclusion | 12 | 5 | 2 | 1 | 2 | 12 |
| **Total** | | **96** | **32** | **25** | **12** | **21** | **~70** |

Note: Some blocks combine into single notebook cells (e.g., a computation block and its immediate narrative may share one cell). The total of ~70 cells reflects this merging, with approximately 55–75 actual cells depending on how code+output and table+visualization pairs are combined.

---

## Implementation Notes

Memory management: The raw CSV is ~119MB (2.3M rows). After preprocessing, the data.pt file is 301MB. Use float32 (not float64) for tensors to cut memory in half. Use .copy() sparingly; prefer in-place operations or views where safe. For PCA on 30K subsamples, the full test set is not needed in memory simultaneously.

GPU/CPU fallback: Check torch.cuda.is_available() before sending tensors to GPU. AMD ROCm 7.2 on RX 9060 XT is the primary GPU. If CUDA/ROCm unavailable, all operations fall back to CPU — MLP training will be slower (expect ~5x longer), KNN cdist will use torch on CPU (still functional but slower), XGBoost will also fall back to CPU (set device='cpu' accordingly). Set device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') at notebook startup.

ROCm compatibility: PyTorch ROCm build required (not standard CUDA PyTorch). Verify with torch.cuda.is_available() and torch.cuda.get_device_name(0). If "ROCm" or "AMD" appears in device name, GPU is correctly configured. Mixed precision (AMP) works on ROCm via torch.cuda.amp.GradScaler and autocast. No NVIDIA-specific extensions used.

Chronological split: This is the single most critical design decision. Shuffling time series data causes look-ahead bias — the model learns future patterns that would not be available in real deployment. All preprocessing steps that involve fitting parameters (StandardScaler, PCA) must fit on training set only to prevent information leakage from validation/test into training.

Confusion matrix warning: The confusion matrix in Section 5.8–5.9 evaluates DIRECTION ONLY (sign of prediction vs sign of actual). This is NOT a classification model — it is a derived binary view of regression outputs. Precision/Recall/F1 for "Down" and "Up" classes should not be interpreted as standalone classifier performance. The baseline is ~50% (random coin flip on balanced data). Any score near 50% does not mean the model is broken — 1-min forex direction is inherently near-random.

Data pair consistency across sections: All notebooks cells must use USD/CHF as the currency pair. File path: data/processed/USDCHF_1min_2020_2026.csv. Prior template blocks may have referenced other pairs (EURUSD) — those references must be updated to USD/CHF in the implementation.

XGBoost GPU (ROCm): XGBoost 3.1.1 from ROCm/xgboost fork. Use tree_method='hist', device='cuda'. Note: device='hip' is NOT a valid param — always use device='cuda' which maps to HIP at compile time. GPU training verified with EllpackDMatrix on device. Grid search and full training both run on GPU.

MLP weight initialization: Use default PyTorch initialization for Linear layers (Kaiming uniform). Do not manually set seeds per layer — the global seed=42 at notebook top is sufficient and keeps the implementation clean.

Output directories: Create outputs/preprocessed/, outputs/models/, outputs/plots/ with os.makedirs(exist_ok=True) at notebook startup to avoid file-not-found errors mid-pipeline.
