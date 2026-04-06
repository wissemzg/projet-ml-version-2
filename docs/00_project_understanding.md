# Project Understanding — IHEC-CODELAB 2.0

## Business Goal (BVMT Context)
Build an **Intelligent Trading Assistant** for the **Bourse des Valeurs Mobilières de Tunis (BVMT)** — the Tunisian stock exchange. The system must help investors make informed decisions, detect suspicious market activity in near-real-time, and provide transparent, explainable recommendations compliant with CMF (Conseil du Marché Financier) regulations.

### Market Specificities
- Variable liquidity across BVMT stocks
- Multilingual information sources (French + Arabic)
- Need for market manipulation surveillance
- Regulatory transparency requirements (CMF)

## Modules Required

### Module 1 — Forecasting (Price + Liquidity)
- **Inputs**: Historical BVMT data (OHLCV — open/high/low/close/volume), technical indicators
- **Outputs**: 5-day closing price forecast, daily volume prediction, liquidity probability (high/low)
- **Metrics**: RMSE, MAE, Directional Accuracy
- **Approach**: SARIMA/SARIMAX baseline → LSTM/XGBoost → ensemble

### Module 2 — Sentiment Analysis (NLP)
- **Inputs**: Scraped Tunisian financial news (French + Arabic) from 3+ sources
- **Outputs**: Daily sentiment score per company (positive/negative/neutral), aggregated score
- **Approach**: Pretrained multilingual models (CamemBERT for French, AraBERT for Arabic)

### Module 3 — Anomaly Detection (Market Surveillance)
- **Inputs**: OHLCV data, transaction counts, computed features
- **Outputs**: Alerts for volume spikes (>3σ), abnormal price variations (>5%), suspicious patterns
- **Metrics**: Precision, Recall, F1
- **Approach**: Statistical z-score + Isolation Forest + rule-based pattern detection

### Module 4 — Decision & Portfolio Management
- **Inputs**: Forecasts, sentiment, anomaly flags, user risk profile
- **Outputs**: Buy/Sell/Hold recommendations, portfolio simulation (ROI, Sharpe, Max Drawdown)
- **Approach**: Rule-based policy with explainability (SHAP-like feature attribution)

## Mandatory UI Pages
1. **Market Overview**: TUNINDEX, top gainers/losers, global sentiment, alerts
2. **Stock Detail**: Historical chart, 5-day forecast + intervals, sentiment, RSI/MACD, recommendation + explanation
3. **My Portfolio**: Positions, allocation, performance, suggestions
4. **Surveillance & Alerts**: Real-time-ish anomaly feed, filters, history

## Evaluation Metrics
| Module | Metrics |
|--------|---------|
| Forecasting | RMSE, MAE, Directional Accuracy |
| Sentiment | Accuracy, F1 per class |
| Anomaly | Precision, Recall, F1 |
| Portfolio | ROI, Sharpe Ratio, Max Drawdown |

## Data Sources
- `histo_cotation_2022-2025.csv`: BVMT historical quotations (SEANCE, GROUPE, CODE, VALEUR, OUVERTURE, CLOTURE, PLUS_BAS, PLUS_HAUT, QUANTITE_NEGOCIEE, NB_TRANSACTION, CAPITAUX)
- `web_histo_cotation_2022-2023.csv`: Web-scraped quotation data (same schema)

## Assumptions (labeled)
- **[ASSUMPTION]** No live BVMT API available → use historical CSV data with simulated streaming
- **[ASSUMPTION]** Intraday data not available at hourly granularity → adapt anomaly detection to daily granularity
- **[ASSUMPTION]** News scraping will use simulated/cached data for demo robustness
- **[ASSUMPTION]** API key provided is for OpenAI GPT-4o access for agent chat
