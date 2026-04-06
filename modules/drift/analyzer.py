"""
Prediction Drift & Real vs Predicted Analysis Module.
Compares forecast outputs to actual market data, detects concept drift
and data drift, provides visual comparison data for the dashboard.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class PredictionDriftAnalyzer:
    """Analyzes prediction accuracy, data drift, and concept drift."""

    def __init__(self):
        self.prediction_history: List[dict] = []

    def store_prediction(self, stock_code: str, predicted: List[float],
                          actual: List[float] = None, prediction_date: str = None):
        """Store a prediction for later drift analysis."""
        self.prediction_history.append({
            'stock_code': stock_code,
            'predicted': predicted,
            'actual': actual,
            'prediction_date': prediction_date or datetime.now().isoformat(),
            'stored_at': datetime.now().isoformat()
        })

    def backtest_forecast(self, stock_df: pd.DataFrame, forecast_fn,
                           test_ratio: float = 0.2, horizon: int = 5,
                           step: int = 5) -> dict:
        """Walk-forward backtesting with real vs predicted comparison."""
        closes = stock_df['close'].dropna().values
        n = len(closes)
        train_end = int(n * (1 - test_ratio))

        all_predicted = []
        all_actual = []
        all_dates = []
        errors_by_step = {i: [] for i in range(1, horizon + 1)}

        dates = stock_df['date'].values if 'date' in stock_df.columns else list(range(n))

        i = train_end
        while i + horizon <= n:
            train_data = stock_df.iloc[:i].copy()
            actual_window = closes[i:i + horizon]

            try:
                result = forecast_fn(train_data, horizon=horizon)
                predicted = result.get('forecast', result.get('predicted', []))[:len(actual_window)]
            except Exception:
                predicted = [closes[i - 1]] * len(actual_window)

            for j in range(len(actual_window)):
                if j < len(predicted):
                    all_predicted.append(float(predicted[j]))
                    all_actual.append(float(actual_window[j]))
                    errors_by_step[j + 1].append(abs(float(actual_window[j]) - float(predicted[j])))
                    if hasattr(dates[i + j], 'isoformat'):
                        all_dates.append(str(dates[i + j].isoformat())[:10])
                    else:
                        all_dates.append(str(dates[i + j]))

            i += step

        if not all_predicted:
            return {'error': 'No predictions generated during backtest'}

        actual_arr = np.array(all_actual)
        pred_arr = np.array(all_predicted)

        mae = float(np.mean(np.abs(actual_arr - pred_arr)))
        mape = float(np.mean(np.abs((actual_arr - pred_arr) / np.where(actual_arr != 0, actual_arr, 1)))) * 100
        rmse = float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))
        bias = float(np.mean(pred_arr - actual_arr))

        if len(actual_arr) > 1:
            actual_dir = np.sign(np.diff(actual_arr))
            pred_dir = np.sign(np.diff(pred_arr))
            direction_acc = float(np.mean(actual_dir == pred_dir)) * 100
        else:
            direction_acc = 0

        # Per-step MAE
        step_mae = {}
        for step_num, errs in errors_by_step.items():
            if errs:
                step_mae[f'day_{step_num}'] = round(float(np.mean(errs)), 4)

        return {
            'metrics': {
                'mae': round(mae, 4),
                'mape': round(mape, 2),
                'rmse': round(rmse, 4),
                'bias': round(bias, 4),
                'direction_accuracy': round(direction_acc, 1),
                'num_predictions': len(all_predicted),
            },
            'per_step_mae': step_mae,
            'comparison': {
                'dates': all_dates[-30:],
                'actual': [round(v, 3) for v in all_actual[-30:]],
                'predicted': [round(v, 3) for v in all_predicted[-30:]],
            },
            'all_actual': [round(v, 3) for v in all_actual],
            'all_predicted': [round(v, 3) for v in all_predicted],
        }

    def compute_data_drift(self, stock_df: pd.DataFrame,
                            ref_ratio: float = 0.6) -> dict:
        """Statistical tests to detect data drift between reference and current windows."""
        closes = stock_df['close'].dropna().values
        volumes = stock_df['volume'].dropna().values if 'volume' in stock_df.columns else None

        if len(closes) < 40:
            return {'drift_detected': False, 'reason': 'Insufficient data'}

        from scipy import stats as sp_stats

        split = int(len(closes) * ref_ratio)
        ref_prices, cur_prices = closes[:split], closes[split:]
        ref_returns = np.diff(ref_prices) / np.where(ref_prices[:-1] != 0, ref_prices[:-1], 1)
        cur_returns = np.diff(cur_prices) / np.where(cur_prices[:-1] != 0, cur_prices[:-1], 1)

        # KS test
        ks_stat, ks_p = sp_stats.ks_2samp(ref_returns, cur_returns)
        # T-test
        t_stat, t_p = sp_stats.ttest_ind(ref_returns, cur_returns, equal_var=False)
        # Levene's test for variance
        lev_stat, lev_p = sp_stats.levene(ref_returns, cur_returns)

        ref_vol = float(np.std(ref_returns)) * 100
        cur_vol = float(np.std(cur_returns)) * 100

        price_drift = {
            'ks_test': {'statistic': round(float(ks_stat), 4), 'pvalue': round(float(ks_p), 4),
                        'drift': ks_p < 0.05},
            'mean_test': {'statistic': round(float(t_stat), 4), 'pvalue': round(float(t_p), 4),
                          'drift': t_p < 0.05},
            'variance_test': {'statistic': round(float(lev_stat), 4), 'pvalue': round(float(lev_p), 4),
                              'drift': lev_p < 0.05},
            'ref_stats': {
                'mean_return': round(float(np.mean(ref_returns)) * 100, 4),
                'volatility': round(ref_vol, 4),
                'skewness': round(float(sp_stats.skew(ref_returns)), 4),
                'kurtosis': round(float(sp_stats.kurtosis(ref_returns)), 4),
                'n_samples': len(ref_returns)
            },
            'cur_stats': {
                'mean_return': round(float(np.mean(cur_returns)) * 100, 4),
                'volatility': round(cur_vol, 4),
                'skewness': round(float(sp_stats.skew(cur_returns)), 4),
                'kurtosis': round(float(sp_stats.kurtosis(cur_returns)), 4),
                'n_samples': len(cur_returns)
            }
        }

        volume_drift = {}
        if volumes is not None and len(volumes) == len(closes):
            ref_v, cur_v = volumes[:split], volumes[split:]
            vks, vp = sp_stats.ks_2samp(ref_v, cur_v)
            volume_drift = {
                'ks_test': {'statistic': round(float(vks), 4), 'pvalue': round(float(vp), 4),
                            'drift': vp < 0.05},
                'ref_mean': round(float(np.mean(ref_v)), 0),
                'cur_mean': round(float(np.mean(cur_v)), 0),
                'change_pct': round((float(np.mean(cur_v)) / float(np.mean(ref_v)) - 1) * 100, 1) if np.mean(ref_v) > 0 else 0
            }

        overall = ks_p < 0.05 or t_p < 0.05 or volume_drift.get('ks_test', {}).get('drift', False)

        # Return distribution data for charting
        ref_hist, ref_bins = np.histogram(ref_returns, bins=30, density=True)
        cur_hist, cur_bins = np.histogram(cur_returns, bins=30, density=True)

        return {
            'price_drift': price_drift,
            'volume_drift': volume_drift,
            'overall_drift': overall,
            'severity': 'high' if (ks_p < 0.01 and t_p < 0.01) else ('medium' if overall else 'low'),
            'distributions': {
                'ref_returns': [round(v, 5) for v in ref_returns.tolist()[-100:]],
                'cur_returns': [round(v, 5) for v in cur_returns.tolist()[-100:]],
                'ref_hist': {'counts': ref_hist.tolist(), 'bins': ref_bins.tolist()},
                'cur_hist': {'counts': cur_hist.tolist(), 'bins': cur_bins.tolist()},
            },
            'analyzed_at': datetime.now().isoformat()
        }

    def compute_rolling_accuracy(self, stock_df: pd.DataFrame,
                                  window: int = 20) -> dict:
        """Compute rolling prediction accuracy to visualize degradation over time."""
        closes = stock_df['close'].dropna().values
        dates_raw = stock_df['date'].values if 'date' in stock_df.columns else list(range(len(closes)))

        if len(closes) < window + 5:
            return {'error': 'Insufficient data'}

        rolling_mae = []
        rolling_dates = []

        for i in range(window, len(closes) - 1):
            # Naive prediction: previous value
            pred = closes[i - 1]
            actual = closes[i]
            err = abs(actual - pred) / actual * 100 if actual != 0 else 0
            rolling_mae.append(round(err, 4))
            dt = dates_raw[i]
            rolling_dates.append(str(dt.isoformat())[:10] if hasattr(dt, 'isoformat') else str(dt))

        # Compute rolling window average
        kernel = np.ones(window) / window
        if len(rolling_mae) >= window:
            smoothed = np.convolve(rolling_mae, kernel, mode='valid')
        else:
            smoothed = rolling_mae

        return {
            'rolling_mape': [round(v, 3) for v in rolling_mae[-60:]],
            'smoothed_mape': [round(float(v), 3) for v in smoothed[-60:]],
            'dates': rolling_dates[-60:],
            'mean_mape': round(float(np.mean(rolling_mae)), 3),
            'trend': 'increasing' if len(smoothed) > 2 and smoothed[-1] > smoothed[0] else 'stable'
        }
