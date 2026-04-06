# User Stories & Test Scenarios — BVMT Trading Assistant

## Core User Stories (from Cahier des Charges)

### US-01: Individual Investor — Market Overview
**As** an individual investor,
**I want** to see the overall market performance on a single dashboard,
**So that** I can quickly assess whether the market is bullish or bearish today.

**Acceptance Criteria:**
- [ ] TUNINDEX value and daily change displayed prominently
- [ ] Top 5 gainers and top 5 losers visible
- [ ] Volume summary for the market
- [ ] Page loads in < 3 seconds

**Test Steps:**
1. Navigate to `/` (Market Overview)
2. Verify TUNINDEX chart renders with correct latest date
3. Verify top movers tables show 5 rows each
4. Verify data matches latest CSV entries

---

### US-02: Individual Investor — Stock Forecast
**As** an individual investor,
**I want** to see a 5-day price forecast for a specific stock (e.g., BIAT),
**So that** I can decide whether to enter or exit a position.

**Acceptance Criteria:**
- [ ] Forecast chart shows historical prices + 5-day prediction
- [ ] Confidence interval (95%) band displayed
- [ ] Forecast method identified (SARIMA, XGBoost, or ensemble)
- [ ] Forecast loads in < 10 seconds

**Test Steps:**
1. Navigate to `/stock/BIAT`
2. Click "Lancer Prévision"
3. Verify forecast API returns 200
4. Verify chart updates with predicted values and CI band
5. Verify forecast direction matches model output

---

### US-03: Portfolio Manager — 5000 TND Suggestion
**As** a portfolio manager with 5000 TND to invest,
**I want** the system to suggest an optimal allocation,
**So that** I maximize returns within my risk tolerance.

**Acceptance Criteria:**
- [ ] System generates allocation across 3-5 stocks
- [ ] Each suggestion includes: stock, quantity, reason
- [ ] Risk profile (Conservative/Moderate/Aggressive) affects output
- [ ] Total allocation ≤ 5000 TND
- [ ] Suggestions are explainable (signal breakdown shown)

**Test Steps:**
1. Navigate to `/portfolio`
2. Set budget = 5000, profile = Moderate
3. Click "Obtenir Suggestions"
4. Verify response includes stock list with quantities
5. Verify sum of (price × quantity) ≤ 5000
6. Verify each suggestion has a reason/justification

---

### US-04: Risk Analyst — Anomaly Detection
**As** a risk analyst at the CMF,
**I want** to review detected anomalies in market activity,
**So that** I can investigate potential market manipulation.

**Acceptance Criteria:**
- [ ] Alert list with severity (CRITICAL/HIGH/MEDIUM)
- [ ] Each alert shows: stock, date, type, description
- [ ] Filter by severity and date range
- [ ] Drill-down to individual stock anomaly history

**Test Steps:**
1. Navigate to `/surveillance`
2. Verify anomaly table loads with recent detections
3. Filter by "CRITICAL" → verify only critical alerts shown
4. Click on a stock code → verify drill-down works

---

## Additional User Stories (Edge Cases)

### US-05: Low Liquidity Stock
**As** an investor looking at a micro-cap stock with < 10 daily transactions,
**I want** the system to warn me about low liquidity,
**So that** I don't enter a position I cannot exit.

**Acceptance Criteria:**
- [ ] Liquidity warning displayed when avg daily volume < 1000
- [ ] Forecast confidence interval wider for illiquid stocks
- [ ] Decision engine reduces buy signal strength for illiquid stocks

**Test Steps:**
1. Find a stock with CODE having < 1000 avg daily volume
2. Navigate to `/stock/<CODE>`
3. Verify liquidity warning banner appears
4. Request forecast → verify wider CI than blue-chip stocks
5. Request portfolio suggestion → verify stock excluded or downweighted

---

### US-06: Conflicting Sentiment vs Price Trend
**As** a trader monitoring SFBT,
**I want** the system to flag when news sentiment is strongly positive but price is declining,
**So that** I can investigate the divergence.

**Acceptance Criteria:**
- [ ] Divergence alert when sentiment > 0.5 and 5-day return < -3%
- [ ] Alert visible on both stock detail page and surveillance page
- [ ] Decision engine adjusts recommendation to reflect conflict

**Test Steps:**
1. Simulate positive news for SFBT while price data shows decline
2. Navigate to `/stock/SFBT`
3. Verify divergence warning is displayed
4. Check `/surveillance` for corresponding alert
5. Verify decision engine outputs "HOLD" or reduced confidence

---

### US-07: Missing News Data
**As** a user requesting sentiment analysis for an obscure stock,
**I want** the system to gracefully handle missing news,
**So that** I still get useful recommendations based on other signals.

**Acceptance Criteria:**
- [ ] When no news found, sentiment score defaults to 0 (neutral)
- [ ] UI displays "No recent news available" message
- [ ] Recommendation still generated from technical + forecast signals
- [ ] Sentiment weight redistributed to other signals (0% → others grow proportionally)

**Test Steps:**
1. Request analysis for a stock with no simulated news
2. Verify sentiment section shows "Aucune actualité récente"
3. Verify recommendation is still generated
4. Verify recommendation explanation excludes sentiment reasoning

---

### US-08: Suspicious Anomaly Without Confirmation
**As** a CMF regulator,
**I want** to distinguish between confirmed and unconfirmed anomalies,
**So that** I don't raise false alarms.

**Acceptance Criteria:**
- [ ] Anomalies detected by multiple methods (z-score + Isolation Forest) marked as "CONFIRMED"
- [ ] Single-method detections marked as "POTENTIAL"
- [ ] Confirmed anomalies escalated to CRITICAL severity
- [ ] False positive estimate shown per detection type

**Test Steps:**
1. Inject a data point with unusual volume but normal price
2. Run anomaly detection
3. Verify it's marked as "POTENTIAL" (only volume z-score triggers)
4. Inject a data point with unusual volume AND price AND Isolation Forest flag
5. Verify it's marked as "CONFIRMED"

---

### US-09: Portfolio Constraint — Maximum Position Size
**As** a conservative investor,
**I want** the system to enforce a maximum 20% allocation per stock,
**So that** my portfolio remains diversified.

**Acceptance Criteria:**
- [ ] No single stock exceeds max_position_pct of total portfolio value
- [ ] If recommendation would breach limit, quantity is reduced
- [ ] Warning displayed when approaching limit
- [ ] Different limits per risk profile (Conservative: 20%, Moderate: 30%, Aggressive: 40%)

**Test Steps:**
1. Create portfolio with 5000 TND
2. Try to buy a single stock worth > 20% of portfolio
3. Verify system either blocks or warns about concentration risk
4. Switch to Aggressive profile → verify limit increases to 40%

---

### US-10: Multiple Day Forecast Accuracy Tracking
**As** a data scientist evaluating the system,
**I want** to see backtest results comparing forecasted vs actual prices,
**So that** I can assess model reliability.

**Acceptance Criteria:**
- [ ] Backtest results available via API `/api/forecast/<code>`
- [ ] Metrics displayed: MAE, RMSE, MAPE, Directional Accuracy
- [ ] Visual overlay of predicted vs actual on chart
- [ ] Per-model breakdown (SARIMA vs XGBoost vs Ensemble)

**Test Steps:**
1. Run forecast for BIAT with historical data
2. Verify response includes `metrics` object
3. Verify MAE < 2% of average price for liquid stocks
4. Verify directional accuracy > 50% (better than random)

---

### US-11: Chat Agent — Market Q&A
**As** an investor using the chat interface,
**I want** to ask natural language questions about the market,
**So that** I can get quick answers without navigating dashboards.

**Acceptance Criteria:**
- [ ] Chat responds to questions like "Quel est le cours de BIAT ?"
- [ ] Responses include data-backed information
- [ ] Chat history maintained during session
- [ ] Fallback response when API is unavailable

**Test Steps:**
1. Navigate to `/chat`
2. Type "Quel est le cours de BIAT ?"
3. Verify response includes recent price data
4. Type "Quelles sont les anomalies du jour ?"
5. Verify response references anomaly detection results
6. Disconnect network → verify fallback message appears

---

### US-12: System Safety — Dangerous Operations Blocked
**As** a system administrator,
**I want** the agent system to block dangerous operations,
**So that** the system remains secure.

**Acceptance Criteria:**
- [ ] SafetyGuard blocks commands containing: `rm -rf`, `DROP TABLE`, `os.system`
- [ ] Blocked commands are logged with timestamp and reason
- [ ] Alert raised to admin on repeated dangerous attempts
- [ ] Agent system continues operating after blocking

**Test Steps:**
1. Submit code containing `os.system('rm -rf /')` to execution agent
2. Verify SafetyGuard blocks execution
3. Verify log entry created in `logs/agent.jsonl`
4. Submit normal code after → verify system still functional

---

## Test Matrix Summary

| Test ID | Story | Module | Priority | Automated |
|---------|-------|--------|----------|-----------|
| T-01 | US-01 | Web/Data | P0 | Yes |
| T-02 | US-02 | Forecasting | P0 | Yes |
| T-03 | US-03 | Portfolio | P0 | Yes |
| T-04 | US-04 | Anomaly | P0 | Yes |
| T-05 | US-05 | Portfolio/Data | P1 | Yes |
| T-06 | US-06 | Sentiment/Anomaly | P1 | Partial |
| T-07 | US-07 | Sentiment | P1 | Yes |
| T-08 | US-08 | Anomaly | P1 | Yes |
| T-09 | US-09 | Portfolio | P2 | Yes |
| T-10 | US-10 | Forecasting | P2 | Yes |
| T-11 | US-11 | Chat/Agent | P1 | Partial |
| T-12 | US-12 | Agent/Safety | P0 | Yes |

## Running Tests
```bash
# Unit tests
python -m pytest tests/ -v

# Integration test (requires running app)
python -m pytest tests/ -v -m integration

# Coverage report
python -m pytest tests/ --cov=modules --cov=agents --cov-report=html
```
