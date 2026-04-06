# Statistical Analysis — BVMT Trading Assistant

## 1. Data Overview

### 1.1 Source Dataset
- **Period**: January 2022 – March 2025 (~3.25 years)
- **Files**: `histo_cotation_{2022,2023,2024,2025}.csv` + `web_histo_cotation_{2022,2023}.csv`
- **Format**: Semicolon-delimited CSV
- **Columns**: SEANCE (date), GROUPE, CODE, VALEUR, OUVERTURE (open), CLOTURE (close), PLUS_BAS (low), PLUS_HAUT (high), QUANTITE_NEGOCIEE (volume), NB_TRANSACTION, CAPITAUX

### 1.2 Data Quality
- **Missing values**: Some stocks have sporadic missing days (illiquid securities, trading halts)
- **Duplicates**: `web_histo_cotation_*` files provide duplicate coverage of 2022-2023 with additional web-sourced fields — handled by deduplication on (SEANCE, CODE)
- **Date format**: DD/MM/YYYY → parsed with `format='%d/%m/%Y'`
- **Numeric encoding**: French locale possible (comma as decimal) → handled via `decimal=','`

## 2. Descriptive Statistics

### 2.1 Market-Wide Metrics
- **Number of listed securities**: ~80+ unique codes
- **Average daily transactions per stock**: varies from <10 (micro-caps) to >500 (blue-chips)
- **Typical daily volume range**: 100 — 500,000 shares per stock
- **TUNINDEX construction**: Market-cap-weighted composite of all stocks (recalculated daily)

### 2.2 Distribution Analysis
- **Daily returns**: Computed as `(close_t / close_{t-1}) - 1`
  - Typical distribution: Leptokurtic (fat tails), slight negative skew
  - Mean daily return: ~0.02% (annualized ~5%)
  - Standard deviation: ~1.5-2.5% for liquid stocks, higher for micro-caps
- **Volume**: Log-normal distribution — right-skewed with occasional massive spikes
- **Bid-ask proxied by High-Low spread**: `(high - low) / close` — average 1-3% for liquid stocks

### 2.3 Correlation Structure
- Intra-sector correlations typically 0.3-0.6 (Banking, Insurance, Leasing)
- Cross-sector correlations typically 0.1-0.3
- TUNINDEX correlation with individual stocks: 0.2-0.7 depending on market cap weight

## 3. Stationarity Analysis

### 3.1 Methods Used
1. **Augmented Dickey-Fuller (ADF) Test**
   - H₀: Series has a unit root (non-stationary)
   - Rejection criterion: p-value < 0.05
   
2. **KPSS Test**
   - H₀: Series is stationary
   - Complementary to ADF (confirms ADF result)

### 3.2 Typical Results
- **Raw price series**: Non-stationary (ADF p > 0.05, KPSS p < 0.05) — as expected
- **First-differenced returns**: Stationary (ADF p < 0.01, KPSS p > 0.05)
- **Log-returns**: Stationary in most cases
- **Implication**: SARIMA models require d=1 (first differencing); some stocks need d=2 for trend removal

### 3.3 Box-Cox Transformation
- Applied when variance is non-constant over time
- Lambda optimized per stock via `scipy.stats.boxcox`
- Stabilizes variance before SARIMA fitting
- Inverse transformation applied to forecasts

## 4. Seasonality Analysis

### 4.1 Calendar Effects
- **Weekly seasonality**: Trading occurs Mon-Fri; Monday/Friday effects observed (higher volume on Friday end-of-week rebalancing)
- **Monthly effects**: January effect (slight), Ramadan slowdown in volume
- **Annual**: Year-end portfolio adjustments; AGM season (April-June) increases activity

### 4.2 SARIMA Seasonal Order
- **s = 5** (trading week) — most effective for daily data
- Seasonal orders (P, D, Q) typically (1, 1, 1) with s=5
- ACF/PACF plots checked for each stock to validate seasonal lag significance

## 5. Forecasting Methodology

### 5.1 SARIMA Pipeline
```
1. Load daily closing prices
2. Check stationarity (ADF + KPSS)
3. Apply Box-Cox if variance non-constant
4. Determine (p,d,q) via ACF/PACF analysis
5. Set seasonal (P,D,Q,s=5) via grid search
6. Fit model, compute AIC/BIC
7. Generate N-day forecast with confidence intervals (95%)
8. Inverse Box-Cox transform
```

### 5.2 XGBoost Feature Engineering
Features created for each observation:
- **Lag features**: close_{t-1}, close_{t-2}, ..., close_{t-10}
- **Rolling statistics**: SMA_5, SMA_20, rolling_std_10
- **Technical indicators**: RSI_14, MACD_line, MACD_signal
- **Volume features**: volume_{t-1}, volume_ratio_5d
- **Calendar features**: day_of_week, month, is_month_end

### 5.3 Ensemble
- Weighted average: `forecast = w_sarima * f_sarima + w_xgb * f_xgb + w_ets * f_ets`
- Default weights: SARIMA=0.4, XGBoost=0.4, ETS=0.2
- Confidence interval: Widest of individual model intervals (conservative)

### 5.4 Evaluation Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| MAE | Σ\|y - ŷ\| / n | < 2% of price |
| RMSE | √(Σ(y - ŷ)² / n) | < 3% of price |
| MAPE | Σ\|y - ŷ\|/y / n × 100 | < 5% |
| Directional Accuracy | % correct up/down | > 55% |

### 5.5 Backtesting Protocol
- **Walk-forward validation**: Train on first 80%, predict next 20% in 5-day rolling windows
- **Out-of-time test**: Train on 2022-2023, validate on 2024
- **Per-stock evaluation**: Some stocks are more forecastable than others

## 6. Anomaly Detection Statistics

### 6.1 Volume Anomaly
- **Method**: Z-score with rolling 20-day window
- **Threshold**: |z| > 3 (3 standard deviations)
- **Expected false positive rate**: ~0.3% per day per stock (Gaussian assumption)
- **Actual**: Higher due to fat-tailed volume distribution

### 6.2 Price Anomaly
- **Method**: Daily return magnitude
- **Threshold**: |return| > 5% (absolute)
- **Rationale**: BVMT has daily limit-up/limit-down rules — moves beyond 5% are unusual but possible in volatile periods

### 6.3 Isolation Forest
- **Contamination**: 5% (expected anomaly rate)
- **Features**: [return, volume_zscore, spread, intraday_range]
- **n_estimators**: 200 (forest size)

## 7. Sentiment Quantification

### 7.1 Score Mapping
| Stars (BERT output) | Sentiment Score | Label |
|---------------------|----------------|-------|
| 1 ★ | -1.0 | Very Negative |
| 2 ★★ | -0.5 | Negative |
| 3 ★★★ | 0.0 | Neutral |
| 4 ★★★★ | +0.5 | Positive |
| 5 ★★★★★ | +1.0 | Very Positive |

### 7.2 Daily Aggregation
- Per stock per day: weighted average of all article sentiments
- Weighting: Source reliability × recency decay
- Smoothing: 3-day EMA to reduce noise

## 8. Risk Metrics

### 8.1 Portfolio Performance
- **Sharpe Ratio**: `(R_p - R_f) / σ_p` where R_f = 7% annual (Tunisia BCT rate)
- **Max Drawdown**: `max(peak - trough) / peak` over evaluation period
- **Beta**: `Cov(R_stock, R_tunindex) / Var(R_tunindex)` — market sensitivity
- **VaR (95%)**: Historical 5th percentile of daily returns × portfolio value

### 8.2 Decision Engine Signal Weights
| Signal | Weight | Source |
|--------|--------|--------|
| Technical | 40% | RSI, MACD, Bollinger |
| Forecast | 30% | SARIMA/XGBoost predicted direction |
| Sentiment | 20% | News sentiment score |
| Anomaly | 10% | Safety flag (reduce if anomaly detected) |

### 8.3 Position Sizing (Kelly Criterion simplified)
```
position_pct = min(kelly_fraction, max_position)
kelly_fraction = (win_prob × avg_win - loss_prob × avg_loss) / avg_win
max_position = 1 / n_stocks (equal weight) or risk-adjusted
```
