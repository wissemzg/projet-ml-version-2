# Model Choices — BVMT Trading Assistant

## Module 1 — Forecasting

### Candidate Models
| Model | Type | Pros | Cons |
|-------|------|------|------|
| SARIMA | Statistical | Interpretable, handles seasonality, no GPU needed | Limited to linear patterns |
| SARIMAX | Statistical | Adds exogenous variables | Same linear limitation |
| ETS (Holt-Winters) | Statistical | Good for trend + seasonal | Cannot capture complex dependencies |
| XGBoost | ML (tree) | Fast, handles nonlinear, feature importance | No sequential memory |
| LSTM | Deep Learning | Captures long sequences | Needs more data, GPU preferred |

### Final Pick
- **Primary**: SARIMA/SARIMAX baseline (interpretable, CMF-compliant)
- **Secondary**: XGBoost with engineered features (best accuracy/speed tradeoff)
- **Ensemble**: Weighted average of available models

### Justification
- **Dataset size**: ~4 years of daily data (~80K records across all stocks) — sufficient for statistical + ML, borderline for deep learning
- **Compute**: Local CPU — XGBoost preferred over LSTM for latency
- **Interpretability**: CMF requires transparent recommendations → SARIMA provides clear parametric interpretation
- **Latency**: Sub-second for SARIMA/XGBoost; LSTM would add 2-5s per stock

### Hyperparameter Strategy
- SARIMA: Grid search over (p,d,q) × (P,D,Q,s) with AIC minimization; s=5 (trading week)
- XGBoost: n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8

## Module 2 — Sentiment

### Candidate Models
| Model | Language | Pros | Cons |
|-------|----------|------|------|
| CamemBERT | French | Best French NLP model | French only |
| AraBERT | Arabic | Best Arabic NLP model | Arabic only |
| mBERT-sentiment | Multilingual | Covers both languages | Slightly less accurate per language |
| Keyword-based | Both | Fast, no model needed | Low accuracy |

### Final Pick
- **Primary**: `nlptown/bert-base-multilingual-uncased-sentiment` (5-star rating → sentiment mapping)
- **Fallback**: Custom keyword dictionary (French + Arabic financial terms)

### Justification
- **Language needs**: French + Arabic required → multilingual model is the only option that handles both
- **Compute**: Transformer inference is ~100ms per text on CPU — acceptable for batch processing
- **Training**: Using pretrained model (no fine-tuning needed given data constraints)
- **Accuracy**: Acceptable for market sentiment aggregation (precision matters more than recall)

## Module 3 — Anomaly Detection

### Candidate Models
| Model | Type | Pros | Cons |
|-------|------|------|------|
| Z-score | Statistical | Simple, interpretable, fast | Univariate only |
| Isolation Forest | ML | Multivariate, unsupervised | Less interpretable |
| DBSCAN | Clustering | Density-based detection | Sensitive to parameters |
| Autoencoder | Deep Learning | Complex pattern detection | Needs large data, hard to explain |

### Final Pick
- **Statistical**: Z-score for volume spikes (>3σ) + percentage threshold for price anomalies (>5%)
- **ML**: Isolation Forest for multivariate anomaly scoring
- **Rule-based**: Custom pattern detectors (volume-price divergence, consecutive extremes, EOD manipulation)

### Justification
- **Interpretability**: Market surveillance requires explainable alerts → rule-based + z-score
- **Coverage**: Isolation Forest catches patterns missed by rules → used as complement
- **No labels**: Anomaly detection is unsupervised → Isolation Forest ideal
- **Speed**: All methods sub-second per stock

## Module 4 — Decision & Portfolio

### Approach
- **Rule-based policy** (minimum viable, no RL)
- **Weighted signal aggregation**: Forecast (30%) + Technical (40%) + Sentiment (20%) + Anomaly (10%)
- **Risk profiles**: Conservative/Moderate/Aggressive with different thresholds

### Justification
- **Explainability**: MANDATORY per cahier des charges → rule-based provides clear signal attribution
- **RL considered but rejected**: Insufficient labeled reward data; RL would be opaque to CMF
- **Risk management**: Stop-loss, position sizing, diversification rules built in

## Tradeoffs Summary

| Concern | Conservative Choice | Aggressive Choice | Our Pick |
|---------|--------------------|--------------------|----------|
| Accuracy | SARIMA | LSTM ensemble | XGBoost + SARIMA |
| Speed | Keywords | Transformer | Multilingual BERT (batched) |
| Explainability | Rule-based | Black-box ML | Rule-based + SHAP-like attribution |
| Compute | CPU only | GPU cluster | CPU (local deployment) |
| Cost | Free models | Commercial APIs | Pretrained open-source + OpenAI for chat |
