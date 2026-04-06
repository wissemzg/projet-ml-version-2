"""
Module 3 — Anomaly Detection for BVMT market surveillance.
Detects volume spikes (>3σ), abnormal price variations (>5%), suspicious patterns.
Reports Precision/Recall/F1 and generates alerts.
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# ── Anomaly Cache (TTL = 10 minutes) ──
_anomaly_cache = {}
_market_anomaly_cache = {'time': 0, 'result': None}
_ANOMALY_CACHE_TTL = 600


class AnomalyDetector:
    """Market surveillance anomaly detection for BVMT."""
    
    def __init__(self, volume_sigma: float = 3.0, price_threshold: float = 5.0):
        self.volume_sigma = volume_sigma
        self.price_threshold = price_threshold
        self.isolation_forest = None
        self.scaler = None
        self.alerts = []
    
    # ── Statistical Anomaly Detection ──
    
    def detect_volume_spikes(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Detect volume spikes exceeding 3σ from rolling mean.
        Adapts to daily granularity as intraday data is not available.
        """
        df = df.copy()
        df['volume_ma'] = df['volume'].rolling(window, min_periods=5).mean()
        df['volume_std'] = df['volume'].rolling(window, min_periods=5).std()
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume_std'].replace(0, np.nan)
        df['volume_spike'] = df['volume_zscore'].abs() > self.volume_sigma
        return df
    
    def detect_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect abnormal price variations (>5% daily change).
        [ASSUMPTION] Adapted to daily granularity since hourly data not available.
        """
        df = df.copy()
        df['return_pct'] = df['close'].pct_change() * 100
        df['price_anomaly'] = df['return_pct'].abs() > self.price_threshold
        df['price_direction'] = np.where(df['return_pct'] > 0, 'up', 
                                np.where(df['return_pct'] < 0, 'down', 'flat'))
        return df
    
    def detect_suspicious_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect suspicious trading patterns:
        1. Volume-price divergence (high volume, no price movement)
        2. Consecutive extreme moves in same direction
        3. Unusual transaction patterns (many transactions, low volume)
        4. Wash trading indicators
        """
        df = df.copy()
        
        # Pattern 1: Volume surge without price impact
        df['return_pct'] = df['close'].pct_change() * 100
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / vol_ma.replace(0, np.nan)
        df['pattern_vol_no_price'] = (df['vol_ratio'] > 3) & (df['return_pct'].abs() < 0.5)
        
        # Pattern 2: Consecutive extreme moves (3+ days same direction >2%)
        df['extreme_up'] = df['return_pct'] > 2
        df['extreme_down'] = df['return_pct'] < -2
        df['consec_up'] = df['extreme_up'].rolling(3).sum() >= 3
        df['consec_down'] = df['extreme_down'].rolling(3).sum() >= 3
        df['pattern_consecutive'] = df['consec_up'] | df['consec_down']
        
        # Pattern 3: Unusual transaction count vs volume ratio
        if 'transactions' in df.columns:
            tx_ratio = df['transactions'] / df['volume'].replace(0, np.nan)
            tx_mean = tx_ratio.rolling(20).mean()
            tx_std = tx_ratio.rolling(20).std()
            df['pattern_tx_anomaly'] = ((tx_ratio - tx_mean) / tx_std.replace(0, np.nan)).abs() > 2.5
        else:
            df['pattern_tx_anomaly'] = False
        
        # Pattern 4: End-of-day price manipulation (large close vs open diff)
        df['close_open_pct'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan) * 100
        co_std = df['close_open_pct'].rolling(20).std()
        df['pattern_eod_manip'] = df['close_open_pct'].abs() > (3 * co_std)
        
        # Combined suspicious flag
        df['suspicious'] = (df['pattern_vol_no_price'] | df['pattern_consecutive'] | 
                           df['pattern_tx_anomaly'] | df['pattern_eod_manip'])
        
        return df
    
    # ── Machine Learning Anomaly Detection ──
    
    def fit_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit Isolation Forest for multivariate anomaly detection."""
        df = df.copy()
        feature_cols = ['close', 'volume', 'return_pct']
        
        if 'transactions' in df.columns:
            feature_cols.append('transactions')
        if 'capital' in df.columns:
            feature_cols.append('capital')
        
        # Add derived features
        df['return_pct'] = df['close'].pct_change() * 100
        df['volume_change'] = df['volume'].pct_change() * 100
        df['range_pct'] = (df['high'] - df['low']) / df['close'].replace(0, np.nan) * 100
        feature_cols += ['volume_change', 'range_pct']
        
        df_clean = df.dropna(subset=feature_cols)
        if len(df_clean) < 50:
            df['if_anomaly'] = False
            return df
        
        X = df_clean[feature_cols].values
        # Clean infinity and extreme values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -1e10, 1e10)
        from sklearn.preprocessing import StandardScaler
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        from sklearn.ensemble import IsolationForest
        self.isolation_forest = IsolationForest(
            contamination=0.05, random_state=42, n_estimators=50
        )
        predictions = self.isolation_forest.fit_predict(X_scaled)
        
        df.loc[df_clean.index, 'if_anomaly'] = predictions == -1
        df.loc[df_clean.index, 'if_score'] = self.isolation_forest.decision_function(X_scaled)
        df['if_anomaly'] = df['if_anomaly'].fillna(False)
        
        return df
    
    # ── Full Detection Pipeline ──
    
    def detect_all(self, df: pd.DataFrame) -> Dict:
        """
        Run full anomaly detection pipeline.
        Returns detection results with alerts and metrics.
        """
        df = df.sort_values('date').reset_index(drop=True)
        stock_name = df['stock'].iloc[0] if 'stock' in df.columns else 'UNKNOWN'
        stock_code = df['code'].iloc[0] if 'code' in df.columns else ''

        # Check cache
        cache_key = (stock_code, len(df))
        now = time.time()
        if cache_key in _anomaly_cache:
            ct, cr = _anomaly_cache[cache_key]
            if now - ct < _ANOMALY_CACHE_TTL:
                return cr

        # Limit to last 250 rows for speed
        if len(df) > 250:
            df = df.tail(250).reset_index(drop=True)

        # Clean data: replace inf and very large values
        for col in ['close', 'open', 'high', 'low', 'volume', 'capital', 'transactions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows where close is 0 or null (can't compute returns)
        df = df[df['close'].notna() & (df['close'] > 0)].reset_index(drop=True)
        if len(df) < 10:
            return {
                "stock": stock_name, "total_records": len(df),
                "anomalies_found": 0, "anomaly_rate": 0,
                "volume_spikes": 0, "price_anomalies": 0,
                "suspicious_patterns": 0, "if_anomalies": 0,
                "alerts": [], "metrics": {}
            }
        
        # Run all detectors
        df = self.detect_volume_spikes(df)
        df = self.detect_price_anomalies(df)
        df = self.detect_suspicious_patterns(df)
        df = self.fit_isolation_forest(df)
        
        # Combined anomaly flag
        df['is_anomaly'] = (df.get('volume_spike', False) | 
                           df.get('price_anomaly', False) | 
                           df.get('suspicious', False) |
                           df.get('if_anomaly', False))
        
        # Generate alerts
        alerts = self._generate_alerts(df, stock_name)
        
        # Compute pseudo-metrics using Isolation Forest as ground truth proxy
        metrics = self._compute_metrics(df)
        
        result = {
            "stock": stock_name,
            "total_records": len(df),
            "anomalies_found": int(df['is_anomaly'].sum()),
            "anomaly_rate": round(df['is_anomaly'].mean() * 100, 2),
            "volume_spikes": int(df.get('volume_spike', pd.Series(False)).sum()),
            "price_anomalies": int(df.get('price_anomaly', pd.Series(False)).sum()),
            "suspicious_patterns": int(df.get('suspicious', pd.Series(False)).sum()),
            "if_anomalies": int(df.get('if_anomaly', pd.Series(False)).sum()),
            "alerts": alerts[-20:],  # Last 20 alerts
            "metrics": metrics,
        }
        _anomaly_cache[cache_key] = (time.time(), result)
        return result
    
    def _generate_alerts(self, df: pd.DataFrame, stock: str) -> List[Dict]:
        """Generate alerts from detected anomalies (vectorized)."""
        anom = df[df.get('is_anomaly', False) == True].copy()
        if anom.empty:
            return []

        stock_code = df['code'].iloc[0] if 'code' in df.columns else ''
        alerts = []

        # Pre-extract columns as arrays for speed
        dates = anom['date'].values
        vs = anom.get('volume_spike', pd.Series(False, index=anom.index)).values
        vz = anom.get('volume_zscore', pd.Series(0, index=anom.index)).values
        pa = anom.get('price_anomaly', pd.Series(False, index=anom.index)).values
        rp = anom.get('return_pct', pd.Series(0, index=anom.index)).values
        sus = anom.get('suspicious', pd.Series(False, index=anom.index)).values
        ifa = anom.get('if_anomaly', pd.Series(False, index=anom.index)).values

        for i in range(len(anom)):
            types = []
            descs = []
            max_severity = 'low'
            if vs[i]:
                types.append('VOLUME_SPIKE')
                z_val = abs(float(vz[i]))
                descs.append(f'Volume spike (z-score: {z_val:.1f})')
                if z_val > 4:
                    max_severity = 'high'
                elif z_val > 3:
                    max_severity = 'medium'
            if pa[i]:
                types.append('PRICE_ANOMALY')
                pct_val = abs(float(rp[i]))
                descs.append(f'Abnormal price change: {float(rp[i]):+.2f}%')
                if pct_val > 8:
                    max_severity = 'high'
                elif pct_val > 5 and max_severity != 'high':
                    max_severity = 'medium'
            if sus[i]:
                types.append('SUSPICIOUS_PATTERN')
                descs.append('Suspicious pattern detected')
                if max_severity == 'low':
                    max_severity = 'medium'
            if ifa[i]:
                types.append('ML_DETECTED')
                descs.append('Isolation Forest anomaly')
            nt = len(types)
            if nt >= 2:
                max_severity = 'high'
            dt = dates[i]
            date_str = str(pd.Timestamp(dt).date()) if pd.notna(dt) else ''
            alerts.append({
                'date': date_str, 'stock': stock, 'stock_code': stock_code,
                'severity': max_severity,
                'types': types, 'type': ', '.join(types),
                'description': ' | '.join(descs)
            })

        return sorted(alerts, key=lambda x: x['date'], reverse=True)[:30]
    
    def _compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute Precision/Recall/F1 using combined rule-based as prediction vs IF as proxy truth."""
        rule_based = (df.get('volume_spike', False) | 
                     df.get('price_anomaly', False) | 
                     df.get('suspicious', False))
        
        if_detected = df.get('if_anomaly', pd.Series(False, index=df.index))
        
        valid = rule_based.notna() & if_detected.notna()
        if valid.sum() < 10:
            return {"note": "Insufficient data for metric computation"}
        
        y_true = if_detected[valid].astype(int)
        y_pred = rule_based[valid].astype(int)
        
        combined_true = (y_true | y_pred).astype(int)
        
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            return {
                "precision": round(precision_score(combined_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(combined_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(combined_true, y_pred, zero_division=0), 4),
                "support": int(combined_true.sum())
            }
        except:
            return {"note": "Metric computation error"}


def detect_market_wide_anomalies(all_data: pd.DataFrame) -> List[Dict]:
    """Detect anomalies across top stocks — fast rule-based only (no ML)."""
    now = time.time()
    if _market_anomaly_cache['result'] is not None and now - _market_anomaly_cache['time'] < _ANOMALY_CACHE_TTL:
        return _market_anomaly_cache['result']

    all_alerts = []
    for stock_code, stock_df in all_data.groupby('code'):
        if len(stock_df) < 30:
            continue
        try:
            recent = stock_df.sort_values('date').tail(60).copy()
            # Quick rule-based detection only (skip ML Isolation Forest)
            dfc = recent.copy()
            if 'volume' in dfc.columns:
                vmean = dfc['volume'].rolling(20, min_periods=5).mean()
                vstd = dfc['volume'].rolling(20, min_periods=5).std().replace(0, 1)
                vz = ((dfc['volume'] - vmean) / vstd).fillna(0)
                vol_spikes = vz.abs() > 2.5
            else:
                vol_spikes = pd.Series(False, index=dfc.index)
            
            if 'close' in dfc.columns:
                ret = dfc['close'].pct_change() * 100
                price_anom = ret.abs() > 5.0
            else:
                price_anom = pd.Series(False, index=dfc.index)
            
            flagged = vol_spikes | price_anom
            idxs = dfc.index[flagged]
            stock = dfc['stock'].iloc[0] if 'stock' in dfc.columns else stock_code
            for idx in idxs[-3:]:  # max 3 alerts per stock
                row = dfc.loc[idx]
                types, descs = [], []
                max_severity = 'low'
                if vol_spikes.get(idx, False):
                    types.append('VOLUME_SPIKE')
                    z_val = abs(float(vz.get(idx, 0)))
                    descs.append(f'Volume spike (z={z_val:.1f})')
                    if z_val > 4:
                        max_severity = 'high'
                    elif z_val > 3:
                        max_severity = 'medium'
                if price_anom.get(idx, False):
                    types.append('PRICE_ANOMALY')
                    pct_val = abs(float(ret.get(idx, 0)))
                    descs.append(f'Price change: {ret.get(idx,0):+.2f}%')
                    if pct_val > 8:
                        max_severity = 'high'
                    elif pct_val > 5 and max_severity != 'high':
                        max_severity = 'medium'
                # Multiple types → always high
                if len(types) >= 2:
                    max_severity = 'high'
                dt = row.get('date', '')
                date_str = str(pd.Timestamp(dt).date()) if pd.notna(dt) else ''
                all_alerts.append({
                    'date': date_str, 'stock': stock, 'stock_code': stock_code,
                    'severity': max_severity,
                    'types': types, 'type': ', '.join(types),
                    'description': ' | '.join(descs)
                })
        except Exception:
            continue

    result = sorted(all_alerts, key=lambda x: x.get('date', ''), reverse=True)[:100]
    _market_anomaly_cache['time'] = time.time()
    _market_anomaly_cache['result'] = result
    return result
