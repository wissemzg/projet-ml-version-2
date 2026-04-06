"""
Module 1 — Forecasting Engine for BVMT stocks.
Fast models: EMA/Linear Regression extrapolation + XGBoost ensemble.
No SARIMA — it hangs on Windows due to thread unkillable nature.
All computations complete in <2s per stock.
"""
import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Forecast Cache (TTL = 30 minutes) ──
_forecast_cache = {}
_CACHE_TTL = 1800


class BVMTForecaster:
    """Price & volume forecasting for BVMT stocks — fast models only."""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.price_model = None
        self.volume_model = None
        self.scaler = None
        self.metrics = {}

    def _get_scaler(self):
        if self.scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        return self.scaler

    @staticmethod
    def _price_floor(last_price: float) -> float:
        """Never return negative/zero prices in forecasts."""
        try:
            lp = float(last_price)
        except Exception:
            lp = 0.0
        return max(0.001, lp * 0.05)

    @staticmethod
    def _clamp_forecast(values, floor: float):
        out = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                fv = floor
            if np.isnan(fv) or np.isinf(fv):
                fv = floor
            out.append(max(floor, fv))
        return out

    # ── Stationarity & Diagnostics ──

    @staticmethod
    def compute_aic_bic(residuals, n_params: int) -> dict:
        """
        Compute AIC and BIC from residuals using Gaussian log-likelihood.
        AIC = 2k - 2*ln(L)
        BIC = k*ln(n) - 2*ln(L)
        where L is the maximized likelihood, k is number of parameters, n is sample size.
        """
        residuals = np.asarray(residuals, dtype=float)
        residuals = residuals[~np.isnan(residuals)]
        n = len(residuals)
        if n < 5:
            return {"aic": None, "bic": None}
        
        sse = float(np.sum(residuals ** 2))
        sigma2 = sse / n
        if sigma2 <= 0:
            sigma2 = 1e-10
        
        # Gaussian log-likelihood: -n/2 * (ln(2*pi) + ln(sigma2) + 1)
        log_likelihood = -n / 2.0 * (np.log(2 * np.pi) + np.log(sigma2) + 1.0)
        
        k = n_params
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        return {
            "aic": round(float(aic), 2),
            "bic": round(float(bic), 2),
            "log_likelihood": round(float(log_likelihood), 2),
            "n_params": k,
            "n_obs": n,
            "sigma2": round(float(sigma2), 6)
        }

    @staticmethod
    def compute_forecast_metrics(actual, predicted) -> dict:
        """Compute RMSE, MAE, and Directional Accuracy."""
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        n = min(len(actual), len(predicted))
        if n < 2:
            return {}
        actual = actual[:n]
        predicted = predicted[:n]
        
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae = float(np.mean(np.abs(actual - predicted)))
        
        # Directional accuracy
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        da = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) > 0 else 0
        
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "directional_accuracy": round(da, 2)
        }

    @staticmethod
    def check_stationarity(series: pd.Series, name: str = "series") -> dict:
        """Quick ADF stationarity test (limited data for speed)."""
        clean = series.dropna().tail(100)
        if len(clean) < 20:
            return {"stationary": False, "reason": "Too few observations"}
        try:
            from statsmodels.tsa.stattools import adfuller
            _adf_result = adfuller(clean, maxlag=5, autolag=None)
            adf_stat, adf_p = float(_adf_result[0]), float(_adf_result[1])
        except Exception:
            return {"name": name, "stationary": False, "adf_pvalue": 1.0}
        return {
            "name": name,
            "adf_statistic": round(adf_stat, 4),
            "adf_pvalue": round(adf_p, 4),
            "adf_stationary": adf_p < 0.05,
            "stationary": adf_p < 0.05,
            "recommendation": "Stationary" if adf_p < 0.05 else "Difference needed"
        }

    # ── Feature Engineering ──

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for ML model."""
        feat = df.copy()
        for lag in [1, 2, 3, 5, 10]:
            feat[f'close_lag_{lag}'] = feat['close'].shift(lag)
            feat[f'volume_lag_{lag}'] = feat['volume'].shift(lag)
        feat['close_ma5'] = feat['close'].rolling(5).mean()
        feat['close_ma10'] = feat['close'].rolling(10).mean()
        feat['close_ma20'] = feat['close'].rolling(20).mean()
        feat['volume_ma5'] = feat['volume'].rolling(5).mean()
        feat['return_1d'] = feat['close'].pct_change()
        feat['return_5d'] = feat['close'].pct_change(5)
        feat['volatility_5'] = feat['return_1d'].rolling(5).std()
        feat['volatility_10'] = feat['return_1d'].rolling(10).std()
        feat['day_of_week'] = feat['date'].dt.dayofweek
        feat['month'] = feat['date'].dt.month
        feat['high_low_range'] = feat['high'] - feat['low']
        feat['open_close_diff'] = feat['close'] - feat['open']
        return feat.dropna()

    # ── EMA Extrapolation (replaces SARIMA) ──

    def fit_ema_forecast(self, series: pd.Series, horizon: int = 5) -> dict | None:
        """
        Exponential Moving Average extrapolation — instant, no fitting.
        Uses EMA(10) trend + mean reversion damping. ~0ms.
        """
        data = series.dropna().tail(100).values
        if len(data) < 10:
            return None
        # EMA-10 and EMA-20
        def _ema(arr, span):
            alpha = 2.0 / (span + 1)
            out = np.empty_like(arr, dtype=float)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
            return out

        ema10 = _ema(data, 10)
        ema20 = _ema(data, 20)
        last_price = float(data[-1])
        floor = self._price_floor(last_price)
        # Trend = slope of EMA10 over last 5 days
        trend = (ema10[-1] - ema10[-6]) / 5 if len(ema10) >= 6 else 0
        # Damping factor: trend decays toward 0 over the horizon
        forecasts = []
        for i in range(1, horizon + 1):
            damping = max(0, 1 - i * 0.15)  # decay 15% per day
            pred = last_price + trend * i * damping
            forecasts.append(round(float(max(floor, pred)), 3))

        # Confidence interval from recent volatility
        returns = np.diff(np.asarray(data, dtype=float)) / np.asarray(data[:-1], dtype=float)
        vol = float(np.std(returns)) * last_price if len(returns) > 5 else last_price * 0.01
        lower = [round(f - 1.96 * vol * np.sqrt(i + 1), 3) for i, f in enumerate(forecasts)]
        upper = [round(f + 1.96 * vol * np.sqrt(i + 1), 3) for i, f in enumerate(forecasts)]

        lower = self._clamp_forecast(lower, floor)
        upper = self._clamp_forecast(upper, floor)

        return {
            "forecast": forecasts,
            "lower_ci": lower,
            "upper_ci": upper,
            "method": "EMA_extrapolation",
            "ema10": round(float(ema10[-1]), 3),
            "ema20": round(float(ema20[-1]), 3),
            "trend_per_day": round(float(trend), 4),
        }

    # ── Linear Regression Forecast (replaces ETS) ──

    def fit_linear_forecast(self, series: pd.Series, horizon: int = 5) -> dict | None:
        """
        Weighted linear regression on last 30 points — instant.
        More recent points get higher weight.
        """
        data = np.asarray(series.dropna().tail(30).values, dtype=float)
        if len(data) < 10:
            return None
        last_price = float(data[-1])
        floor = self._price_floor(last_price)
        n = len(data)
        x = np.arange(n, dtype=float)
        # Exponential weights: more weight on recent
        weights = np.exp(np.linspace(-1, 0, n))
        # Weighted least squares
        w_sum = weights.sum()
        xm = np.average(x, weights=weights)
        ym = np.average(data, weights=weights)
        cov_xy = np.average((x - xm) * (data - ym), weights=weights)
        var_x = np.average((x - xm) ** 2, weights=weights)
        slope = cov_xy / var_x if var_x > 0 else 0
        intercept = ym - slope * xm

        forecasts = []
        for i in range(1, horizon + 1):
            pred = intercept + slope * (n - 1 + i)
            forecasts.append(round(float(max(floor, pred)), 3))

        residuals = data - (intercept + slope * x)
        std_err = float(np.std(residuals))
        lower = [round(f - 1.96 * std_err, 3) for f in forecasts]
        upper = [round(f + 1.96 * std_err, 3) for f in forecasts]

        lower = self._clamp_forecast(lower, floor)
        upper = self._clamp_forecast(upper, floor)

        return {
            "forecast": forecasts,
            "lower_ci": lower,
            "upper_ci": upper,
            "method": "weighted_linear_regression",
            "slope_per_day": round(float(slope), 4),
            "r_squared": round(float(1 - np.var(residuals) / np.var(data)) if np.var(data) > 0 else 0, 4),
        }

    # ── XGBoost Model ──

    def fit_xgboost(self, df: pd.DataFrame, target='close'):
        """Train XGBoost model for price/volume prediction."""
        feat = self.create_features(df.tail(200))
        feature_cols = [c for c in feat.columns if c not in
                       ['date', 'stock', 'code', 'group', 'source_file', target,
                        'open', 'high', 'low', 'close', 'volume', 'capital', 'transactions']]
        feature_cols = [c for c in feature_cols if feat[c].dtype in ['float64', 'int64', 'int32']]
        if len(feature_cols) == 0:
            return None, None

        X = feat[feature_cols].values
        y = feat[target].values
        split = int(len(X) * 0.8)
        if split < 5:
            return None, None
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.15,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method='hist'
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        if len(y_test) > 1:
            actual_dir = np.diff(y_test) > 0
            pred_dir = np.diff(y_pred) > 0
            da = np.mean(actual_dir == pred_dir) * 100
        else:
            da = 0

        return model, {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'directional_accuracy': round(da, 2),
            'features': feature_cols
        }

    # ── Main Forecasting Pipeline ──

    def forecast(self, df: pd.DataFrame, horizon: int = 5, fast: bool = False) -> dict:
        """
        Fast forecasting pipeline (~1-2s total):
        1. Stationarity check (ADF)
        2. EMA extrapolation (instant)
        3. Weighted linear regression (instant)
        4. XGBoost (~1s)
        5. Ensemble average + CI
        """
        # Check cache first
        cache_key = (self.stock_code, len(df), horizon, bool(fast))
        now = time.time()
        if cache_key in _forecast_cache:
            cached_time, cached_result = _forecast_cache[cache_key]
            if now - cached_time < _CACHE_TTL:
                return cached_result

        df = df.sort_values('date').reset_index(drop=True)
        if len(df) > 300:
            df = df.tail(300).reset_index(drop=True)
        close_series = df['close'].dropna()
        if 'volume' in df.columns:
            volume_series = df['volume'].dropna()
        else:
            volume_series = pd.Series(np.zeros(len(df), dtype=float))

        if len(close_series) < 30:
            return {"error": "Insufficient data for forecasting"}

        results = {
            "stock_code": self.stock_code,
            "horizon": horizon,
            "last_date": str(df['date'].max().date()),
            "last_close": float(close_series.iloc[-1]),
        }

        last_price = float(close_series.iloc[-1])
        floor = self._price_floor(last_price)

        # 1. Stationarity diagnostics
        # ADF test is fast (100 points, maxlag=5) — always run it.
        results["stationarity"] = self.check_stationarity(close_series, "close_price")
        results["volume_stationarity"] = self.check_stationarity(volume_series, "volume")

        # 2. EMA extrapolation (replaces SARIMA — instant)
        ema_result = self.fit_ema_forecast(close_series, horizon)
        ema_forecast = []
        if ema_result:
            ema_forecast = ema_result["forecast"]
            # Compute AIC/BIC from EMA fitted residuals
            ema_data = close_series.dropna().tail(100).values
            if len(ema_data) >= 10:
                def _ema_fit(arr, span):
                    alpha = 2.0 / (span + 1)
                    out = np.empty_like(arr, dtype=float)
                    out[0] = arr[0]
                    for i in range(1, len(arr)):
                        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
                    return out
                ema_fitted = _ema_fit(ema_data, 10)
                ema_residuals = ema_data[1:] - ema_fitted[:-1]  # one-step-ahead residuals
                ema_ic = self.compute_aic_bic(ema_residuals, n_params=2)  # span + intercept
            else:
                ema_ic = {"aic": None, "bic": None}
            results["sarima"] = {
                "forecast": ema_result["forecast"],
                "lower_ci": ema_result["lower_ci"],
                "upper_ci": ema_result["upper_ci"],
                "aic": ema_ic.get("aic"),
                "bic": ema_ic.get("bic"),
                "log_likelihood": ema_ic.get("log_likelihood"),
                "method": "ema_extrapolation",
                "ema10": ema_result.get("ema10"),
                "ema20": ema_result.get("ema20"),
                "trend_per_day": ema_result.get("trend_per_day"),
            }

        # 3. Weighted linear regression (replaces ETS — instant)
        lr_result = self.fit_linear_forecast(close_series, horizon)
        lr_forecast = []
        if lr_result:
            lr_forecast = lr_result["forecast"]
            # Compute AIC/BIC from LR residuals
            lr_data = np.asarray(close_series.dropna().tail(30).values, dtype=float)
            if len(lr_data) >= 10:
                n_lr = len(lr_data)
                x_lr = np.arange(n_lr, dtype=float)
                weights_lr = np.exp(np.linspace(-1, 0, n_lr))
                xm_lr = np.average(x_lr, weights=weights_lr)
                ym_lr = np.average(lr_data, weights=weights_lr)
                cov_xy_lr = np.average((x_lr - xm_lr) * (lr_data - ym_lr), weights=weights_lr)
                var_x_lr = np.average((x_lr - xm_lr) ** 2, weights=weights_lr)
                slope_lr = cov_xy_lr / var_x_lr if var_x_lr > 0 else 0
                intercept_lr = ym_lr - slope_lr * xm_lr
                lr_residuals = lr_data - (intercept_lr + slope_lr * x_lr)
                lr_ic = self.compute_aic_bic(lr_residuals, n_params=2)  # slope + intercept
            else:
                lr_ic = {"aic": None, "bic": None}
            results["ets"] = {
                "forecast": lr_result["forecast"],
                "lower_ci": lr_result["lower_ci"],
                "upper_ci": lr_result["upper_ci"],
                "aic": lr_ic.get("aic"),
                "bic": lr_ic.get("bic"),
                "log_likelihood": lr_ic.get("log_likelihood"),
                "method": "weighted_linear_regression",
                "slope_per_day": lr_result.get("slope_per_day"),
                "r_squared": lr_result.get("r_squared"),
            }

        # 4. XGBoost forecast
        xgb_model, xgb_metrics = (None, None)
        if not fast:
            try:
                xgb_model, xgb_metrics = self.fit_xgboost(df, target='close')
            except Exception:
                xgb_model, xgb_metrics = (None, None)
        xgb_forecast = []
        if xgb_model and xgb_metrics:
            self.price_model = xgb_model
            self.metrics['xgboost_price'] = xgb_metrics
            feat = self.create_features(df)
            feature_cols = xgb_metrics['features']
            last_row = feat[feature_cols].iloc[-1:].values
            for i in range(horizon):
                pred = xgb_model.predict(last_row)[0]
                xgb_forecast.append(float(pred))
            results["xgboost"] = {
                "forecast": [round(v, 3) for v in self._clamp_forecast(xgb_forecast, floor)],
                "metrics": xgb_metrics
            }

        # 5. Ensemble forecast (average of available models)
        all_forecasts = []
        weights = []
        if ema_forecast:
            all_forecasts.append(ema_forecast)
            weights.append(0.25)
        if lr_forecast:
            all_forecasts.append(lr_forecast)
            weights.append(0.25)
        if xgb_forecast:
            all_forecasts.append(xgb_forecast)
            weights.append(0.50)  # XGBoost gets more weight (ML model)

        if all_forecasts:
            max_len = horizon
            padded = []
            for f in all_forecasts:
                if len(f) >= max_len:
                    padded.append(f[:max_len])
                else:
                    padded.append(f + [f[-1]] * (max_len - len(f)))

            # Weighted average
            w_total = sum(weights[:len(padded)])
            w_norm = [w / w_total for w in weights[:len(padded)]]
            ensemble = np.zeros(max_len)
            for f, w in zip(padded, w_norm):
                ensemble += np.array(f) * w

            ensemble_std = np.std(padded, axis=0) if len(padded) > 1 else np.ones(max_len) * close_series.std() * 0.1

            results["ensemble"] = {
                "forecast": [round(v, 3) for v in self._clamp_forecast(ensemble.tolist(), floor)],
                "lower_ci": [round(v, 3) for v in self._clamp_forecast([v - 1.96 * s for v, s in zip(ensemble, ensemble_std)], floor)],
                "upper_ci": [round(v, 3) for v in self._clamp_forecast([v + 1.96 * s for v, s in zip(ensemble, ensemble_std)], floor)],
                "num_models": len(all_forecasts)
            }

            last_date = df['date'].max()
            forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
            results["forecast_dates"] = [str(d.date()) for d in forecast_dates]

        # Collect best AIC/BIC across all models
        all_aic = []
        all_bic = []
        for key in ['sarima', 'ets', 'xgboost']:
            model_res = results.get(key, {})
            if model_res.get('aic') is not None:
                all_aic.append(model_res['aic'])
            if model_res.get('bic') is not None:
                all_bic.append(model_res['bic'])
            # Also check metrics sub-dict (xgboost)
            metrics_sub = model_res.get('metrics', {})
            if metrics_sub.get('aic') is not None:
                all_aic.append(metrics_sub['aic'])
            if metrics_sub.get('bic') is not None:
                all_bic.append(metrics_sub['bic'])

        results["model_selection"] = {
            "best_aic": round(min(all_aic), 2) if all_aic else None,
            "best_bic": round(min(all_bic), 2) if all_bic else None,
            "all_aic": {k: results.get(k, {}).get('aic') for k in ['sarima', 'ets'] if results.get(k, {}).get('aic') is not None},
            "all_bic": {k: results.get(k, {}).get('bic') for k in ['sarima', 'ets'] if results.get(k, {}).get('bic') is not None},
        }

        # Compute backtest metrics (RMSE, MAE, DA) from recent data
        try:
            recent = close_series.tail(30).values
            if len(recent) >= 15:
                train_part = recent[:20]
                test_part = recent[20:]
                if len(test_part) >= 5:
                    series_train = pd.Series(train_part)
                    ema_bt = self.fit_ema_forecast(series_train, len(test_part))
                    lr_bt = self.fit_linear_forecast(series_train, len(test_part))
                    pred_bt = []
                    if ema_bt and lr_bt:
                        for j in range(len(test_part)):
                            pred_bt.append((ema_bt['forecast'][j] + lr_bt['forecast'][j]) / 2)
                    elif ema_bt:
                        pred_bt = ema_bt['forecast'][:len(test_part)]
                    elif lr_bt:
                        pred_bt = lr_bt['forecast'][:len(test_part)]
                    if pred_bt:
                        results["backtest_metrics"] = self.compute_forecast_metrics(test_part, pred_bt)
        except Exception:
            pass

        # 6. Volume forecast — simple moving average
        avg_vol = float(volume_series.tail(20).mean())
        results["volume_forecast"] = {
            "forecast": [round(avg_vol, 0)] * horizon,
            "liquidity_probability": [{"high": 0.5, "low": 0.5}] * horizon
        }

        _forecast_cache[cache_key] = (time.time(), results)
        return results

    # ── Evaluation ──

    def evaluate_backtest(self, df: pd.DataFrame, horizon: int = 5) -> dict:
        """Walk-forward backtest using fast EMA+LR models (no SARIMAX)."""
        close = df['close'].dropna().values
        n = len(close)
        if n < 100:
            return {"error": "Insufficient data for backtesting"}

        test_size = min(60, n // 5)
        actuals, predictions = [], []

        for i in range(0, test_size - horizon, 5):
            train_end = n - test_size + i
            train = close[:train_end]
            actual = close[train_end:train_end + horizon]
            if len(actual) < horizon or len(train) < 20:
                continue
            try:
                # Use EMA extrapolation for backtest (same as forecast)
                series = pd.Series(train)
                ema_r = self.fit_ema_forecast(series, horizon)
                lr_r = self.fit_linear_forecast(series, horizon)
                preds = []
                if ema_r and lr_r:
                    for j in range(horizon):
                        preds.append((ema_r['forecast'][j] + lr_r['forecast'][j]) / 2)
                elif ema_r:
                    preds = ema_r['forecast']
                elif lr_r:
                    preds = lr_r['forecast']
                else:
                    preds = [float(train[-1])] * horizon
                actuals.append(actual)
                predictions.append(preds)
            except Exception:
                continue

        if not actuals:
            return {"error": "Backtest failed"}

        actuals = np.array(actuals)
        predictions = np.array(predictions)
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))
        actual_dirs = np.diff(actuals, axis=1) > 0
        pred_dirs = np.diff(predictions, axis=1) > 0
        da = np.mean(actual_dirs == pred_dirs) * 100

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "directional_accuracy": round(da, 2),
            "num_windows": len(actuals)
        }
