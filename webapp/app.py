"""
BVMT Trading Assistant — Main Flask Application (Enhanced)
Full-stack: Market Overview, Stock Detail, Portfolio, Surveillance, Chat,
            Agent Dashboard
CrewAI multi-agent + MCP tools + A2A integration
"""
import sys, os, traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env BEFORE any os.environ.get() calls
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_executor = ThreadPoolExecutor(max_workers=4)

# ── Direct module imports ──
print("  Loading modules...", flush=True)

from modules.common.data_loader import (
    load_all_data, get_stock_data, get_stock_list,
    compute_tunindex, add_technical_indicators
)
from modules.forecasting.forecaster import BVMTForecaster
from modules.anomaly.detector import AnomalyDetector, detect_market_wide_anomalies
from modules.portfolio.manager import PortfolioSimulator, DecisionEngine, RiskProfile
from modules.drift.analyzer import PredictionDriftAnalyzer
from agents.agent_system import ChatAgent

from modules.user.manager import UserManager
from modules.rl.portfolio_rl import PortfolioRL
from modules.scraper.realtime import RealtimeScraper
from modules.portfolio.db import (
    create_portfolio as db_create_portfolio,
    buy_stock as db_buy_stock,
    sell_stock as db_sell_stock,
    get_portfolio as db_get_portfolio,
    get_user_portfolio,
    list_portfolios
)
import time as _time

try:
    from modules.sentiment.analyzer import SentimentAnalyzer, generate_simulated_news
except ImportError:
    SentimentAnalyzer = None
    generate_simulated_news = None

try:
    from agents.crew import OrchestratorAgent, MCPToolRegistry
    _reg = MCPToolRegistry()
    orchestrator = OrchestratorAgent(_reg)
except Exception:
    orchestrator = None

# ── Workflow log (A2A messages) — persists across requests ──
_workflow_log = []

def _add_workflow_entry(sender, receiver, action, detail=""):
    """Append a timestamped A2A workflow entry."""
    _workflow_log.append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'sender': sender,
        'receiver': receiver,
        'action': action,
        'detail': detail,
    })
    # Keep only last 50 entries
    if len(_workflow_log) > 50:
        del _workflow_log[:-50]

print("  [OK] All modules loaded.", flush=True)

# ── Flask app ──
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'bvmt-trading-2025-hackathon'
CORS(app)

# ── Configuration ──
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

# ── Initialize objects ──
_data_cache = {}
portfolios = {}
chat_agent = ChatAgent(api_key=OPENAI_API_KEY)
drift_analyzer = PredictionDriftAnalyzer()
sentiment_analyzer = SentimentAnalyzer(api_key=OPENAI_API_KEY) if (SentimentAnalyzer and OPENAI_API_KEY) else None
user_manager = UserManager()
realtime_scraper = RealtimeScraper(interval_seconds=60)
realtime_scraper.start()  # Start background scraping


@app.before_request
def _enforce_login_and_onboarding():
    """Require login first, then onboarding, before allowing access."""
    path = request.path or ''

    # Public resources
    if path.startswith('/static/'):
        return None
    if path == '/login':
        return None
    if path.startswith('/api/auth/'):
        return None

    uid = session.get('user_id')
    if not uid:
        if path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'auth_required'}), 401
        return redirect(url_for('login_page'))

    # Allow onboarding endpoints/pages when logged in
    if path == '/chat' or path.startswith('/api/user/onboarding') or path.startswith('/api/chat'):
        return None

    try:
        user = user_manager.get_user(uid)
    except Exception:
        user = None

    if user and not bool(getattr(user, 'profile_completed', False)):
        if path.startswith('/api/'):
            return jsonify({'success': False, 'error': 'onboarding_required'}), 403
        return redirect(url_for('chat_page', onboarding=1))

    return None

# ── Pre-load data at startup ──
print("  Loading market data...", flush=True)
_data_cache['all_data'] = load_all_data()
print(f"  [OK] Loaded {len(_data_cache['all_data']):,} records.", flush=True)

# Build a fast lookup index for code -> row indices (avoids full boolean scan per request)
try:
    _data_cache['code_index'] = _data_cache['all_data'].groupby('code').indices
    print(f"  [OK] Built code index for {len(_data_cache['code_index']):,} stocks.", flush=True)

    # Precompute last N rows per stock for ultra-fast per-request access.
    _TAIL_N = 600
    _tail_cache = {}
    _df_all = _data_cache['all_data']
    for _code, _idx in _data_cache['code_index'].items():
        try:
            _tidx = _idx[-_TAIL_N:] if len(_idx) > _TAIL_N else _idx
            _sub = _df_all.iloc[_tidx]
            _tail_cache[_code] = {
                'date': _sub['date'].tolist(),
                'close': pd.to_numeric(_sub['close'], errors='coerce').fillna(0.0).astype(float).tolist(),
                'volume': pd.to_numeric(_sub['volume'], errors='coerce').fillna(0.0).astype(float).tolist() if 'volume' in _sub.columns else [],
            }
        except Exception:
            continue
    _data_cache['tail_cache'] = _tail_cache
    print(f"  [OK] Built tail cache for {len(_tail_cache):,} stocks.", flush=True)
except Exception as _e:
    _data_cache['code_index'] = None
    _data_cache['tail_cache'] = None
    print(f"  [WARN] Could not build code index: {_e}", flush=True)


def get_cached_data():
    return _data_cache['all_data']


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def json_safe(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else v
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif isinstance(obj, float):
        return 0.0 if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


def _get_stock_df_by_code(df: pd.DataFrame, stock_code: str, tail: int = 500) -> pd.DataFrame:
    """Fast path to retrieve one stock's history.

    Uses the precomputed groupby index to avoid O(N) boolean scans.
    """
    idx_map = _data_cache.get('code_index')
    if isinstance(idx_map, dict) and stock_code in idx_map:
        idx = idx_map[stock_code]
        idx = idx[-tail:] if len(idx) > tail else idx
        sub = df.iloc[idx]
        return sub.copy()
    return df[df['code'] == stock_code].sort_values('date').tail(tail).copy()


def _quick_forecast_from_tail_payload(tail_payload: dict, stock_code: str, horizon: int = 5) -> dict:
    dates = tail_payload.get('date') or []
    closes = tail_payload.get('close') or []
    volumes = tail_payload.get('volume') or []

    close = [float(x) for x in closes if x is not None]
    if len(close) < 2:
        return {"error": "Insufficient data for forecasting"}

    last_close = float(close[-1])
    floor = max(0.001, last_close * 0.05)

    lookback = min(6, len(close))
    prev = float(close[-lookback])
    steps = max(1, lookback - 1)
    trend = (last_close - prev) / steps

    forecasts = []
    for i in range(1, horizon + 1):
        damping = max(0.0, 1.0 - i * 0.15)
        pred = last_close + trend * i * damping
        fv = float(pred)
        if np.isnan(fv) or np.isinf(fv):
            fv = last_close
        forecasts.append(round(float(max(floor, fv)), 3))

    tail_close = np.asarray(close[-min(60, len(close)):], dtype=float)
    rets = np.diff(tail_close) / np.maximum(tail_close[:-1], 1e-9)
    vol_amt = float(np.std(rets)) * last_close if len(rets) >= 5 else last_close * 0.01
    lower = []
    upper = []
    for i, f in enumerate(forecasts, start=1):
        spread = 1.96 * vol_amt * np.sqrt(i)
        lo = float(max(floor, f - spread))
        hi = float(max(floor, f + spread))
        lower.append(round(lo, 3))
        upper.append(round(hi, 3))

    last_date = dates[-1] if dates else None
    try:
        last_ts = pd.to_datetime(last_date)
        forecast_dates = pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=horizon)
        forecast_dates = [str(d.date()) for d in forecast_dates]
        last_date_str = str(last_ts.date())
    except Exception:
        forecast_dates = []
        last_date_str = str(last_date) if last_date is not None else ''

    avg_vol = float(np.mean(volumes[-20:])) if volumes else 0.0

    # Run quick ADF stationarity test on close prices
    stationarity_result = {"name": "close_price", "skipped": True}
    try:
        close_arr = np.asarray(close[-min(100, len(close)):], dtype=float)
        if len(close_arr) >= 20:
            from statsmodels.tsa.stattools import adfuller
            _adf_res = adfuller(close_arr, maxlag=5, autolag=None)
            adf_stat, adf_p = float(_adf_res[0]), float(_adf_res[1])
            stationarity_result = {
                "name": "close_price",
                "adf_statistic": round(float(adf_stat), 4),
                "adf_pvalue": round(float(adf_p), 4),
                "adf_stationary": adf_p < 0.05,
                "stationary": adf_p < 0.05,
                "recommendation": "Stationary" if adf_p < 0.05 else "Difference needed"
            }
    except Exception:
        pass

    # Compute AIC/BIC from fitted residuals
    aic_val = None
    bic_val = None
    try:
        if len(close) >= 10:
            n_ic = len(close)
            x_ic = np.arange(n_ic, dtype=float)
            w_ic = np.exp(np.linspace(-1, 0, n_ic))
            xm_ic = np.average(x_ic, weights=w_ic)
            ym_ic = np.average(close, weights=w_ic)
            cov_ic = np.average((x_ic - xm_ic) * (np.array(close) - ym_ic), weights=w_ic)
            var_ic = np.average((x_ic - xm_ic) ** 2, weights=w_ic)
            sl_ic = cov_ic / var_ic if var_ic > 0 else 0
            int_ic = ym_ic - sl_ic * xm_ic
            resid_ic = np.array(close) - (int_ic + sl_ic * x_ic)
            sse_ic = float(np.sum(resid_ic ** 2))
            sigma2_ic = sse_ic / n_ic
            if sigma2_ic > 0:
                ll_ic = -n_ic / 2.0 * (np.log(2 * np.pi) + np.log(sigma2_ic) + 1.0)
                aic_val = round(2 * 2 - 2 * ll_ic, 2)
                bic_val = round(2 * np.log(n_ic) - 2 * ll_ic, 2)
    except Exception:
        pass

    return {
        "stock_code": stock_code,
        "horizon": int(horizon),
        "last_date": last_date_str,
        "last_close": float(last_close),
        "stationarity": stationarity_result,
        "volume_stationarity": {"name": "volume", "skipped": True},
        "sarima": {"aic": aic_val, "bic": bic_val, "method": "quick-damped-trend"},
        "model_selection": {"best_aic": aic_val, "best_bic": bic_val},
        "ensemble": {
            "forecast": forecasts,
            "lower_ci": lower,
            "upper_ci": upper,
            "num_models": 2,
        },
        "forecast_dates": forecast_dates,
        "method": "quick-damped-trend",
        "volume_forecast": {
            "forecast": [round(avg_vol, 0)] * int(horizon),
            "liquidity_probability": [{"high": 0.5, "low": 0.5}] * int(horizon),
        },
    }


def _quick_forecast_from_df(stock_df: pd.DataFrame, stock_code: str, horizon: int = 5) -> dict:
    """Ultra-fast deterministic forecast used for UI responsiveness.

    Guarantees:
    - Always returns quickly (no heavy imports)
    - No NaN/Inf
    - Forecast prices never go <= 0
    """
    sdf = stock_df.sort_values('date').reset_index(drop=True)
    close = sdf['close'].dropna().astype(float)
    if len(close) < 2:
        return {"error": "Insufficient data for forecasting"}

    last_close = float(close.iloc[-1])
    floor = max(0.001, last_close * 0.05)

    # Simple damped trend based on last 5 sessions
    lookback = min(6, len(close))
    prev = float(close.iloc[-lookback])
    steps = max(1, lookback - 1)
    trend = (last_close - prev) / steps

    forecasts = []
    for i in range(1, horizon + 1):
        damping = max(0.0, 1.0 - i * 0.15)
        pred = last_close + trend * i * damping
        fv = float(pred)
        if np.isnan(fv) or np.isinf(fv):
            fv = last_close
        forecasts.append(round(float(max(floor, fv)), 3))

    # CI from recent volatility
    tail_close = np.asarray(close.tail(min(60, len(close))).values, dtype=float)
    rets = np.diff(tail_close) / np.maximum(tail_close[:-1], 1e-9)
    vol_amt = float(np.std(rets)) * last_close if len(rets) >= 5 else last_close * 0.01
    lower = []
    upper = []
    for i, f in enumerate(forecasts, start=1):
        spread = 1.96 * vol_amt * np.sqrt(i)
        lo = float(max(floor, f - spread))
        hi = float(max(floor, f + spread))
        lower.append(round(lo, 3))
        upper.append(round(hi, 3))

    last_date = sdf['date'].max()
    try:
        forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
        forecast_dates = [str(d.date()) for d in forecast_dates]
    except Exception:
        forecast_dates = []

    avg_vol = float(sdf['volume'].dropna().tail(20).mean()) if 'volume' in sdf.columns else 0.0

    # Run quick ADF stationarity test
    stationarity_result = {"name": "close_price", "skipped": True}
    try:
        close_arr = np.asarray(close.tail(min(100, len(close))).values, dtype=float)
        if len(close_arr) >= 20:
            from statsmodels.tsa.stattools import adfuller
            _adf_res = adfuller(close_arr, maxlag=5, autolag=None)
            adf_stat, adf_p = float(_adf_res[0]), float(_adf_res[1])
            stationarity_result = {
                "name": "close_price",
                "adf_statistic": round(float(adf_stat), 4),
                "adf_pvalue": round(float(adf_p), 4),
                "adf_stationary": adf_p < 0.05,
                "stationary": adf_p < 0.05,
                "recommendation": "Stationary" if adf_p < 0.05 else "Difference needed"
            }
    except Exception:
        pass

    # Compute AIC/BIC
    aic_val = None
    bic_val = None
    try:
        close_vals = close.values
        if len(close_vals) >= 10:
            n_ic = len(close_vals)
            x_ic = np.arange(n_ic, dtype=float)
            w_ic = np.exp(np.linspace(-1, 0, n_ic))
            xm_ic = np.average(x_ic, weights=w_ic)
            ym_ic = np.average(close_vals, weights=w_ic)
            cov_ic = np.average((x_ic - xm_ic) * (close_vals - ym_ic), weights=w_ic)
            var_ic = np.average((x_ic - xm_ic) ** 2, weights=w_ic)
            sl_ic = cov_ic / var_ic if var_ic > 0 else 0
            int_ic = ym_ic - sl_ic * xm_ic
            resid_ic = close_vals - (int_ic + sl_ic * x_ic)
            sse_ic = float(np.sum(resid_ic ** 2))
            sigma2_ic = sse_ic / n_ic
            if sigma2_ic > 0:
                ll_ic = -n_ic / 2.0 * (np.log(2 * np.pi) + np.log(sigma2_ic) + 1.0)
                aic_val = round(2 * 2 - 2 * ll_ic, 2)
                bic_val = round(2 * np.log(n_ic) - 2 * ll_ic, 2)
    except Exception:
        pass

    return {
        "stock_code": stock_code,
        "horizon": int(horizon),
        "last_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
        "last_close": float(last_close),
        "stationarity": stationarity_result,
        "volume_stationarity": {"name": "volume", "skipped": True},
        "sarima": {"aic": aic_val, "bic": bic_val, "method": "quick-damped-trend"},
        "model_selection": {"best_aic": aic_val, "best_bic": bic_val},
        "ensemble": {
            "forecast": forecasts,
            "lower_ci": lower,
            "upper_ci": upper,
            "num_models": 2,
        },
        "forecast_dates": forecast_dates,
        "method": "quick-damped-trend",
        "volume_forecast": {
            "forecast": [round(avg_vol, 0)] * int(horizon),
            "liquidity_probability": [{"high": 0.5, "low": 0.5}] * int(horizon),
        },
    }


_sentiment_cache = {}
_SENTIMENT_TTL = 3600  # 1 hour


def get_sentiment_for_stock(stock_name, stock_code=None):
    """Get GPT-4o sentiment analysis for a stock using real price data."""
    cache_key = stock_name
    now = _time.time()
    if cache_key in _sentiment_cache:
        ct, cr = _sentiment_cache[cache_key]
        if now - ct < _SENTIMENT_TTL:
            return cr

    # Build context with real price data
    price_context = ""
    try:
        df = get_cached_data()
        code = stock_code
        if not code:
            match = df[df['stock'] == stock_name]
            if not match.empty:
                code = match['code'].iloc[0]
        if code:
            sdf = df[df['code'] == code].sort_values('date').tail(30)
            if len(sdf) >= 5:
                last = sdf.iloc[-1]
                prev = sdf.iloc[-2]
                chg = ((last['close'] - prev['close']) / prev['close'] * 100) if prev['close'] > 0 else 0
                avg_vol = sdf['volume'].mean()
                vol_chg = ((last['volume'] - avg_vol) / avg_vol * 100) if avg_vol > 0 else 0
                high_30 = sdf['high'].max()
                low_30 = sdf['low'].min()
                returns_5d = ((sdf['close'].iloc[-1] - sdf['close'].iloc[-5]) / sdf['close'].iloc[-5] * 100) if len(sdf) >= 5 else 0
                price_context = (f" Le titre cote actuellement {last['close']:.3f} TND "
                    f"(variation jour: {chg:+.2f}%, variation 5j: {returns_5d:+.2f}%). "
                    f"Volume: {int(last['volume'])} (vs moyenne: {int(avg_vol)}, variation: {vol_chg:+.1f}%). "
                    f"Range 30j: {low_30:.3f} - {high_30:.3f} TND.")
    except Exception:
        pass

    if sentiment_analyzer is not None:
        try:
            text = f"{stock_name} sur la Bourse de Tunis BVMT.{price_context}"
            result = sentiment_analyzer.analyze_text(text)
            score = result.get('score', 0.0)
            sent = result.get('sentiment', 'neutral')
            label = 'Positif' if sent == 'positive' else ('Négatif' if sent == 'negative' else 'Neutre')
            out = {
                'sentiment': label, 'score': round(score, 3),
                'positive_pct': round(max(50 + score * 60, 0), 1),
                'negative_pct': round(max(50 - score * 60, 0), 1),
                'neutral_pct': round(max(0, 100 - max(50 + score * 60, 0) - max(50 - score * 60, 0)), 1),
                'article_count': 1,
                'sources': ['GPT-4o analysis']
            }
            _sentiment_cache[cache_key] = (now, out)
            return out
        except Exception as e:
            print(f"GPT-4o sentiment error for {stock_name}: {e}")
    # Fallback using actual price data if available
    try:
        df = get_cached_data()
        code = stock_code
        if not code:
            match = df[df['stock'] == stock_name]
            if not match.empty:
                code = match['code'].iloc[0]
        if code:
            sdf = df[df['code'] == code].sort_values('date').tail(20)
            if len(sdf) >= 5:
                ret_5 = (sdf['close'].iloc[-1] - sdf['close'].iloc[-5]) / sdf['close'].iloc[-5]
                ret_20 = (sdf['close'].iloc[-1] - sdf['close'].iloc[0]) / sdf['close'].iloc[0]
                vol_ratio = sdf['volume'].iloc[-1] / max(sdf['volume'].mean(), 1)
                score = float(np.clip(ret_5 * 3 + ret_20 * 2 + (vol_ratio - 1) * 0.2, -1, 1))
                label = 'Positif' if score > 0.15 else ('Négatif' if score < -0.15 else 'Neutre')
                out = {
                    'sentiment': label, 'score': round(score, 3),
                    'positive_pct': round(max(50 + score * 60, 0), 1),
                    'negative_pct': round(max(50 - score * 60, 0), 1),
                    'neutral_pct': round(max(10, 100 - max(50 + score * 60, 0) - max(50 - score * 60, 0)), 1),
                    'article_count': 0,
                    'sources': ['price-based analysis']
                }
                _sentiment_cache[cache_key] = (now, out)
                return out
    except Exception:
        pass
    # Last resort fallback
    np.random.seed(hash(stock_name) % 2**32)
    score = np.random.uniform(-0.5, 0.8)
    label = 'Positif' if score > 0.15 else ('Négatif' if score < -0.15 else 'Neutre')
    return {
        'sentiment': label, 'score': round(score, 3),
        'positive_pct': round(max(50 + score * 60, 0), 1),
        'negative_pct': round(max(50 - score * 60, 0), 1),
        'neutral_pct': round(np.random.uniform(5, 25), 1),
        'article_count': 0,
        'sources': ['fallback']
    }


# ══════════════ PAGES ══════════════

@app.route('/login')
def login_page():
    uid = session.get('user_id')
    if uid:
        try:
            user = user_manager.get_user(uid)
        except Exception:
            user = None
        if user and not bool(getattr(user, 'profile_completed', False)):
            return redirect(url_for('chat_page', onboarding=1))
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/')
def index():
    return render_template('market_overview.html')

@app.route('/stock/<stock_code>')
def stock_detail(stock_code):
    return render_template('stock_detail.html', stock_code=stock_code)

@app.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html')

@app.route('/surveillance')
def surveillance_page():
    return render_template('surveillance.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/drift')
def drift_page():
    return render_template('drift.html')

@app.route('/agents')
def agents_page():
    return render_template('agents.html')

@app.route('/trade')
def trade_page():
    return render_template('trade.html')


# ══════════════ API — MARKET ══════════════

_market_cache = {'time': 0, 'data': None}
_MARKET_CACHE_TTL = 300  # 5 minutes


@app.route('/api/market/overview')
def api_market_overview():
    now = _time.time()
    if _market_cache['data'] and now - _market_cache['time'] < _MARKET_CACHE_TTL:
        return jsonify(_market_cache['data'])
    try:
        df = get_cached_data()
        latest_date = df['date'].max()
        prev_date = df[df['date'] < latest_date]['date'].max()

        today = df[df['date'] == latest_date].copy()
        yesterday = df[df['date'] == prev_date].copy()

        merged = today.merge(yesterday[['code', 'close']], on='code', suffixes=('', '_prev'))
        merged['change_pct'] = ((merged['close'] - merged['close_prev']) / merged['close_prev'] * 100).round(2)

        gainers = merged.nlargest(10, 'change_pct')[['code', 'stock', 'close', 'change_pct', 'volume']].to_dict('records')
        losers = merged.nsmallest(10, 'change_pct')[['code', 'stock', 'close', 'change_pct', 'volume']].to_dict('records')

        tunindex_data = compute_tunindex(df)
        ti = tunindex_data['tunindex']
        current_tunindex = round(float(ti.iloc[-1]), 2)
        prev_tunindex = round(float(ti.iloc[-2]), 2) if len(ti) > 1 else current_tunindex
        tunindex_change = round(((current_tunindex / prev_tunindex) - 1) * 100, 2) if prev_tunindex else 0

        total_volume = int(today['volume'].sum())
        total_capital = round(float(today['capital'].sum()), 2)
        num_stocks = int(today['code'].nunique())
        advancing = int((merged['change_pct'] > 0).sum())
        declining = int((merged['change_pct'] < 0).sum())

        hist = tunindex_data[['date', 'tunindex']].tail(90).copy()
        hist['date'] = hist['date'].dt.strftime('%Y-%m-%d')
        hist['tunindex'] = hist['tunindex'].round(2)

        vol_hist = df.groupby('date')['volume'].sum().reset_index().tail(60)
        vol_hist['date'] = vol_hist['date'].dt.strftime('%Y-%m-%d')

        global_sent_score = float(np.mean([get_sentiment_for_stock(s)['score'] for s in today['stock'].unique()[:3]]))

        result = {
            'success': True, 'date': str(latest_date.date()),
            'tunindex': {
                'value': current_tunindex, 'change_pct': tunindex_change,
                'history': hist.to_dict('records')
            },
            'market_stats': {
                'total_volume': total_volume, 'total_capital': total_capital,
                'num_stocks': num_stocks, 'advancing': advancing,
                'declining': declining, 'unchanged': num_stocks - advancing - declining
            },
            'top_gainers': gainers, 'top_losers': losers,
            'volume_history': vol_hist.to_dict('records'),
            'global_sentiment': {
                'score': round(global_sent_score, 3),
                'label': 'Positif' if global_sent_score > 0.15 else ('Négatif' if global_sent_score < -0.15 else 'Neutre')
            },
            'recent_alerts': []
        }
        _market_cache['data'] = result
        _market_cache['time'] = _time.time()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


_stocklist_cache = {'time': 0, 'data': None}
_STOCKLIST_CACHE_TTL = 600  # 10 minutes

@app.route('/api/stocks')
def api_stock_list():
    """Get list of all stocks."""
    try:
        now = _time.time()
        if _stocklist_cache['data'] and now - _stocklist_cache['time'] < _STOCKLIST_CACHE_TTL:
            return jsonify(_stocklist_cache['data'])
        stocks = get_stock_list()
        result = {
            'success': True,
            'stocks': stocks.head(80).assign(
                first_date=lambda x: x['first_date'].dt.strftime('%Y-%m-%d'),
                last_date=lambda x: x['last_date'].dt.strftime('%Y-%m-%d'),
                avg_volume=lambda x: x['avg_volume'].round(0),
                last_close=lambda x: x['last_close'].round(3)
            ).to_dict('records')
        }
        _stocklist_cache['data'] = result
        _stocklist_cache['time'] = now
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


_stock_detail_cache = {}
_STOCK_DETAIL_TTL = 300  # 5 minutes

@app.route('/api/stock/<stock_code>')
def api_stock_detail(stock_code):
    try:
        now_t = _time.time()
        if stock_code in _stock_detail_cache:
            ct, cd = _stock_detail_cache[stock_code]
            if now_t - ct < _STOCK_DETAIL_TTL:
                return jsonify(cd)

        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').copy()
        if stock_df.empty:
            return jsonify({'success': False, 'error': f'Stock {stock_code} not found'})

        stock_name = stock_df['stock'].iloc[-1]
        stock_recent = stock_df.tail(300).copy()
        stock_tech = add_technical_indicators(stock_recent)

        history = stock_tech.tail(120)[['date','open','close','low','high','volume',
                                         'rsi','macd','macd_signal','macd_hist',
                                         'sma_20','sma_50','bb_upper','bb_lower']].copy()
        history['date'] = history['date'].dt.strftime('%Y-%m-%d')
        for c in history.columns:
            if c != 'date':
                history[c] = history[c].apply(lambda x: safe_float(x))

        last_tech = stock_tech.iloc[-1]
        # Use raw data for OHLCV — technical indicators may have NaN for these
        last_raw = stock_df.iloc[-1]
        # Find last non-zero high/low/volume from recent rows
        raw_tail = stock_df.tail(5)
        raw_high = raw_tail['high'].replace(0, np.nan).dropna()
        raw_low = raw_tail['low'].replace(0, np.nan).dropna()
        raw_vol = raw_tail['volume'].replace(0, np.nan).dropna()

        technicals = {k: safe_float(last_tech.get(k)) for k in
                      ['rsi','macd','macd_signal','macd_hist','sma_20','sma_50','volatility_20']}

        # Quick price-based sentiment (no GPT call — instant)
        sdf = stock_df.tail(20)
        if len(sdf) >= 5:
            ret_5 = (sdf['close'].iloc[-1] - sdf['close'].iloc[-5]) / sdf['close'].iloc[-5]
            ret_20 = (sdf['close'].iloc[-1] - sdf['close'].iloc[0]) / sdf['close'].iloc[0]
            score = float(np.clip(ret_5 * 3 + ret_20 * 2, -1, 1))
        else:
            score = 0.0
        score = safe_float(score, 0.0)
        sentiment_label = 'Positif' if score > 0.15 else ('Négatif' if score < -0.15 else 'Neutre')
        sentiment = {
            'sentiment': sentiment_label, 'score': round(float(score), 3),
            'positive_pct': round(safe_float(max(50 + score * 60, 0)), 1),
            'negative_pct': round(safe_float(max(50 - score * 60, 0)), 1),
            'neutral_pct': round(safe_float(max(10, 100 - max(50+score*60,0) - max(50-score*60,0))), 1),
            'article_count': 0, 'sources': ['price-based (fast)']
        }

        # Quick forecast: use last trend (clamped to positive floor)
        closes = stock_df['close'].values[-30:]
        if len(closes) >= 10:
            trend = (closes[-1] - closes[-5]) / 5
            last_p = float(closes[-1])
            floor = max(0.001, last_p * 0.05)
            fc_vals = [round(float(max(floor, closes[-1] + trend * (i+1))), 3) for i in range(5)]
        else:
            last_p = float(closes[-1])
            floor = max(0.001, last_p * 0.05)
            fc_vals = [round(float(max(floor, last_p)), 3)] * 5
        forecast = {
            'forecast': fc_vals,
            'ensemble': {'forecast': fc_vals, 'num_models': 0},
            'forecast_dates': [], 'method': 'trend-fast'
        }

        pct_change = ((fc_vals[-1] - float(closes[-1])) / float(closes[-1]) * 100) if closes[-1] > 0 else 0
        if pct_change > 2:
            action, reason = 'ACHETER', f'Hausse prévue de +{pct_change:.1f}%'
        elif pct_change < -2:
            action, reason = 'VENDRE', f'Baisse prévue de {pct_change:.1f}%'
        else:
            action, reason = 'CONSERVER', f'Variation faible de {pct_change:+.1f}%'

        result = {
            'success': True,
            'stock': {
                'code': stock_code, 'name': stock_name,
                'current_price': safe_float(last_raw['close'], 0),
                'open': safe_float(last_raw['open']),
                'high': safe_float(raw_high.iloc[-1] if len(raw_high) > 0 else last_raw['high']),
                'low': safe_float(raw_low.iloc[-1] if len(raw_low) > 0 else last_raw['low']),
                'volume': int(safe_float(raw_vol.iloc[-1] if len(raw_vol) > 0 else last_raw['volume'])),
                'change_pct': safe_float(last_raw.get('return_pct',
                    ((last_raw['close'] - stock_df.iloc[-2]['close']) / stock_df.iloc[-2]['close'] * 100)
                    if len(stock_df) > 1 and stock_df.iloc[-2]['close'] > 0 else 0))
            },
            'history': history.to_dict('records'),
            'forecast': json_safe(forecast), 'sentiment': sentiment,
            'technicals': technicals,
            'anomalies': [],
            'recommendation': {'action': action, 'reason': reason, 'confidence': 0.5}
        }
        _stock_detail_cache[stock_code] = (_time.time(), result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/stock/<stock_code>/analysis')
def api_stock_analysis(stock_code):
    """Heavy analysis: real forecast + GPT sentiment + anomaly detection (loaded lazily)."""
    try:
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').tail(500).copy()
        if stock_df.empty:
            return jsonify({'success': False, 'error': 'Stock not found'})

        stock_name = stock_df['stock'].iloc[-1]

        def _do_forecast():
            fc = BVMTForecaster(stock_code)
            return fc.forecast(stock_df, horizon=5, fast=True)
        def _do_sentiment():
            return get_sentiment_for_stock(stock_name, stock_code)
        def _do_anomaly():
            det = AnomalyDetector()
            return det.detect_all(stock_df.tail(60).copy())

        futures = {
            _executor.submit(_do_forecast): 'forecast',
            _executor.submit(_do_sentiment): 'sentiment',
            _executor.submit(_do_anomaly): 'anomaly',
        }
        results = {}
        for fut in as_completed(futures, timeout=55):
            results[futures[fut]] = fut.result()

        forecast = results.get('forecast', {})
        sentiment = results.get('sentiment', {'sentiment': 'Neutre', 'score': 0})
        anomaly_result = results.get('anomaly', {})

        last_price = float(stock_df['close'].iloc[-1])
        fc_vals = forecast.get('ensemble', {}).get('forecast', forecast.get('forecast', []))
        fc_last = float(fc_vals[-1]) if fc_vals else last_price
        pct = ((fc_last - last_price) / last_price * 100) if last_price else 0
        if pct > 2:
            action, reason = 'ACHETER', f'Hausse modélisée de +{pct:.1f}%'
        elif pct < -2:
            action, reason = 'VENDRE', f'Baisse modélisée de {pct:.1f}%'
        else:
            action, reason = 'CONSERVER', f'Variation modélisée de {pct:+.1f}%'

        return jsonify(json_safe({
            'success': True,
            'forecast': forecast, 'sentiment': sentiment,
            'anomalies': anomaly_result.get('alerts', [])[:5],
            'recommendation': {'action': action, 'reason': reason, 'confidence': 0.7}
        }))
    except Exception as e:
        return jsonify(json_safe({'success': False, 'error': str(e)}))


@app.route('/api/forecast/<stock_code>')
def api_forecast(stock_code):
    """Ultra-fast forecast endpoint — uses precomputed tail cache only."""
    try:
        horizon = int(request.args.get('horizon', 5))
        horizon = max(1, min(horizon, 30))
        fast = request.args.get('fast', '1').strip().lower() not in ('0', 'false', 'no')

        if fast:
            # ── Primary path: pure-Python on precomputed lists (no pandas) ──
            tail_cache = _data_cache.get('tail_cache')
            if isinstance(tail_cache, dict) and stock_code in tail_cache:
                result = _quick_forecast_from_tail_payload(tail_cache[stock_code], stock_code, horizon=horizon)
                return jsonify(json_safe({'success': True, **result}))
            # Fallback: use code_index to get DataFrame slice
            df = get_cached_data()
            stock_df = _get_stock_df_by_code(df, stock_code, tail=500)
            if stock_df.empty:
                return jsonify({'success': False, 'error': f'Stock {stock_code} not found'}), 404
            result = _quick_forecast_from_df(stock_df, stock_code, horizon=horizon)
            return jsonify(json_safe({'success': True, **result}))
        else:
            df = get_cached_data()
            stock_df = _get_stock_df_by_code(df, stock_code, tail=500)
            if stock_df.empty:
                return jsonify({'success': False, 'error': f'Stock {stock_code} not found'}), 404
            forecaster = BVMTForecaster(stock_code)
            result = forecaster.forecast(stock_df, horizon=horizon, fast=False)
            return jsonify(json_safe({'success': True, **result}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ══════════════ API — DRIFT & COMPARISON ══════════════

_drift_cache = {}
_DRIFT_CACHE_TTL = 600  # 10 minutes

@app.route('/api/drift/<stock_code>')
def api_drift_analysis(stock_code):
    """Full drift analysis: backtest + data drift + rolling accuracy."""
    try:
        now = _time.time()
        if stock_code in _drift_cache:
            ct, cd = _drift_cache[stock_code]
            if now - ct < _DRIFT_CACHE_TTL:
                return jsonify(cd)

        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').tail(300).copy()
        if len(stock_df) < 60:
            return jsonify({'success': False, 'error': 'Données insuffisantes pour l\'analyse de drift'})

        forecaster = BVMTForecaster(stock_code)

        backtest = drift_analyzer.backtest_forecast(
            stock_df, lambda d, h: forecaster.forecast(d, horizon=h, fast=True),
            test_ratio=0.1, horizon=5, step=15
        )
        data_drift = drift_analyzer.compute_data_drift(stock_df)
        rolling = drift_analyzer.compute_rolling_accuracy(stock_df)

        result = json_safe({
            'success': True, 'stock_code': stock_code,
            'backtest': backtest, 'data_drift': data_drift,
            'rolling_accuracy': rolling
        })
        _drift_cache[stock_code] = (_time.time(), result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/drift/compare/<stock_code>')
def api_drift_compare(stock_code):
    """Quick real vs predicted comparison for last N days."""
    try:
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').tail(300).copy()
        n = min(int(request.args.get('days', 10)), 20)

        if len(stock_df) < n + 10:
            return jsonify({'success': False, 'error': 'Données insuffisantes'})

        train = stock_df.iloc[:-n]
        test = stock_df.iloc[-n:]
        forecaster = BVMTForecaster(stock_code)

        # Single forecast call for speed (instead of loop)
        result = forecaster.forecast(train, horizon=n, fast=True)
        ens = result.get('ensemble', {})
        predictions = ens.get('forecast', result.get('forecast', []))
        if not predictions:
            predictions = [float(train['close'].iloc[-1])] * n
        predictions = predictions[:n]

        actual = test['close'].values[:len(predictions)]
        dates = test['date'].dt.strftime('%Y-%m-%d').values[:len(predictions)]

        if len(predictions) == 0 or len(actual) == 0:
            return jsonify({'success': False, 'error': 'Aucune prédiction générée'})

        actual_l = [safe_float(v) for v in actual]
        pred_l = [safe_float(v) for v in predictions]
        errors = [round(abs(a - p), 3) for a, p in zip(actual_l, pred_l)]
        pct_errors = [round(abs(a - p) / a * 100, 2) if a != 0 else 0 for a, p in zip(actual_l, pred_l)]

        return jsonify({
            'success': True, 'stock_code': stock_code,
            'dates': dates.tolist(), 'actual': actual_l, 'predicted': pred_l,
            'errors': errors, 'pct_errors': pct_errors,
            'metrics': {
                'mae': round(float(np.mean(errors)), 3),
                'mape': round(float(np.mean(pct_errors)), 2),
                'rmse': round(float(np.sqrt(np.mean(np.array(errors)**2))), 3),
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# ══════════════ API — AGENTS ══════════════

@app.route('/api/agents/status')
def api_agents_status():
    agents_list = []
    log = list(_workflow_log[-20:])
    if orchestrator is not None:
        agents_list = orchestrator.get_agent_status()
        orch_log = orchestrator.get_workflow_log()[-20:]
        if orch_log:
            log = orch_log
    return jsonify({'success': True, 'agents': agents_list, 'workflow_log': log})


@app.route('/api/agents/analyze/<stock_code>', methods=['POST'])
def api_agents_analyze(stock_code):
    t0 = _time.time()
    steps = []
    try:
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').copy()
        stock_name = stock_df['stock'].iloc[-1] if len(stock_df) > 0 else stock_code

        _add_workflow_entry('Orchestrator', 'ScraperAgent', 'TASK_ASSIGN',
                            f'Analyser {stock_code} ({stock_name})')

        # Step 1: Scraping / Data collection
        t1 = _time.time()
        recent = stock_df.tail(60)
        steps.append({'step': 1, 'name': 'Scraping', 'status': 'completed',
                      'output': f'Collecté {len(stock_df)} enregistrements pour {stock_name}, dernières 60 séances analysées.',
                      'duration_ms': int((_time.time() - t1) * 1000)})
        _add_workflow_entry('ScraperAgent', 'Orchestrator', 'RESULT',
                            f'{len(stock_df)} enregistrements collectés')

        # Step 2: Forecasting
        t2 = _time.time()
        _add_workflow_entry('Orchestrator', 'ForecastAgent', 'TASK_ASSIGN',
                            f'Prévision 5J pour {stock_code}')
        forecaster = BVMTForecaster(stock_code)
        forecast = forecaster.forecast(stock_df, horizon=5, fast=True)
        fc_values = forecast.get('ensemble', {}).get('forecast', forecast.get('forecast', []))
        models = list(forecast.get('ensemble', {}).get('model_weights', {}).keys())
        steps.append({'step': 2, 'name': 'Forecasting', 'status': 'completed',
                      'output': f'Prévision 5 jours: {["{:.3f}".format(v) if isinstance(v,(int,float)) else str(v) for v in (fc_values[:5] if fc_values else [])]} | Modèles: {", ".join(models) if models else "ensemble"}',
                      'duration_ms': int((_time.time() - t2) * 1000)})
        _add_workflow_entry('ForecastAgent', 'Orchestrator', 'RESULT',
                            f'Modèles: {", ".join(models) if models else "ensemble"}')

        # Step 3: Sentiment
        t3 = _time.time()
        _add_workflow_entry('Orchestrator', 'SentimentAgent', 'TASK_ASSIGN',
                            f'Analyse sentiment {stock_name}')
        sentiment = get_sentiment_for_stock(stock_name)
        steps.append({'step': 3, 'name': 'Sentiment', 'status': 'completed',
                      'output': f'Sentiment: {sentiment["sentiment"]} (score: {sentiment["score"]}) — Sources: {", ".join(sentiment.get("sources",[]))}',
                      'duration_ms': int((_time.time() - t3) * 1000)})
        _add_workflow_entry('SentimentAgent', 'Orchestrator', 'RESULT',
                            f'{sentiment["sentiment"]} (score={sentiment["score"]})')

        # Step 4: Anomaly Detection
        t4 = _time.time()
        _add_workflow_entry('Orchestrator', 'AnomalyAgent', 'TASK_ASSIGN',
                            f'Détection anomalies {stock_code}')
        detector = AnomalyDetector()
        anomalies = detector.detect_all(recent)
        alert_list = anomalies.get('alerts', [])
        steps.append({'step': 4, 'name': 'Anomaly', 'status': 'completed',
                      'output': f'{len(alert_list)} anomalies détectées. Types: {", ".join(set(a.get("type","?") for a in alert_list[:10]))}' if alert_list else 'Aucune anomalie détectée.',
                      'duration_ms': int((_time.time() - t4) * 1000)})
        _add_workflow_entry('AnomalyAgent', 'Orchestrator', 'RESULT',
                            f'{len(alert_list)} anomalies détectées')

        # Step 5: Recommendation
        t5 = _time.time()
        _add_workflow_entry('Orchestrator', 'RecommendationAgent', 'TASK_ASSIGN',
                            'Synthèse finale')
        last_price = float(recent['close'].iloc[-1]) if len(recent) > 0 else 0
        fc_last = float(fc_values[-1]) if fc_values else last_price
        pct_change = ((fc_last - last_price) / last_price * 100) if last_price else 0
        if pct_change > 2:
            action = 'ACHETER'
            reason = f'Hausse prévue de +{pct_change:.1f}%'
        elif pct_change < -2:
            action = 'VENDRE'
            reason = f'Baisse prévue de {pct_change:.1f}%'
        else:
            action = 'CONSERVER'
            reason = f'Variation faible de {pct_change:+.1f}%'

        recommendation = {
            'action': action, 'reason': reason,
            'current_price': last_price, 'target_price': fc_last,
            'sentiment': sentiment['sentiment'], 'anomalies_count': len(alert_list)
        }
        steps.append({'step': 5, 'name': 'Recommendation', 'status': 'completed',
                      'output': f'Recommandation: {action} — {reason} | Sentiment: {sentiment["sentiment"]}',
                      'duration_ms': int((_time.time() - t5) * 1000)})
        _add_workflow_entry('RecommendationAgent', 'Orchestrator', 'DECISION',
                            f'{action} — {reason}')

        total_ms = int((_time.time() - t0) * 1000)
        _add_workflow_entry('Orchestrator', 'A2A_Broadcast', 'COMPLETE',
                            f'Analyse {stock_code} terminée en {total_ms}ms')
        return jsonify(json_safe({
            'success': True,
            'steps': steps,
            'data': {
                'forecast': forecast, 'sentiment': sentiment,
                'anomalies': alert_list[:5],
                'recommendation': recommendation,
                'agent_mode': 'direct'
            },
            'duration_ms': total_ms
        }))
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'steps': steps, 'trace': traceback.format_exc()[:500]})


@app.route('/api/agents/scrape', methods=['POST'])
def api_agents_scrape():
    try:
        if orchestrator:
            result = orchestrator.scraper.execute_with_retry('scrape_all', {})
            return jsonify({'success': result.status.value == 'success', 'data': result.data})
        return jsonify({'success': False, 'error': 'CrewAI non disponible'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ══════════════ API — ANOMALIES ══════════════

_anomaly_market_cache = {'time': 0, 'data': None}
_ANOMALY_MARKET_TTL = 600  # 10 minutes

@app.route('/api/anomalies')
def api_anomalies():
    """Get market-wide anomalies/alerts."""
    try:
        now = _time.time()
        if _anomaly_market_cache['data'] and now - _anomaly_market_cache['time'] < _ANOMALY_MARKET_TTL:
            return jsonify(_anomaly_market_cache['data'])
        df = get_cached_data()
        top_stocks = df.groupby('code')['volume'].sum().nlargest(15).index
        filtered = df[df['code'].isin(top_stocks)]
        
        alerts = detect_market_wide_anomalies(filtered)
        result = {
            'success': True,
            'alerts': json_safe(alerts[:50]),
            'total_alerts': len(alerts)
        }
        _anomaly_market_cache['data'] = result
        _anomaly_market_cache['time'] = _time.time()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/anomalies/<stock_code>')
def api_stock_anomalies(stock_code):
    try:
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').tail(250)
        detector = AnomalyDetector()
        result = detector.detect_all(stock_df)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── Portfolio API (SQLite-backed) ──

@app.route('/api/portfolio/create', methods=['POST'])
def api_create_portfolio():
    """Create a new portfolio."""
    data = request.json or {}
    uid = str(session.get('user_id', ''))
    # Default to onboarding profile values when available
    user = None
    try:
        user = user_manager.get_user(uid)
    except Exception:
        user = None
    capital = float(data.get('capital', getattr(user, 'total_capital', 5000) if user else 5000))
    profile = data.get('risk_profile', getattr(user, 'risk_tolerance', 'moderate') if user else 'moderate')
    pid = f"pf_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    result = db_create_portfolio(pid, capital, profile, uid)
    return jsonify(result)


@app.route('/api/portfolio/my')
def api_my_portfolio():
    """Get current user's existing portfolio if any."""
    uid = str(session.get('user_id', ''))
    pid = get_user_portfolio(uid)
    if pid:
        df = get_cached_data()
        latest_date = df['date'].max()
        prices = df[df['date'] == latest_date].set_index('code')['close'].to_dict()
        status = db_get_portfolio(pid, prices)
        return jsonify(status)
    return jsonify({'success': False, 'error': 'No portfolio found'})


@app.route('/api/portfolio/<pid>/buy', methods=['POST'])
def api_portfolio_buy(pid):
    """Buy stock in portfolio."""
    data = request.json
    shares = int(data.get('shares', 0)) if data.get('shares') else None
    price = float(data['price'])
    if not shares and data.get('amount'):
        shares = int(float(data['amount']) / price)
    if not shares:
        shares = 10
    result = db_buy_stock(pid, data['stock_code'], data.get('stock_name', data['stock_code']), price, shares)
    return jsonify(result)


@app.route('/api/portfolio/<pid>/sell', methods=['POST'])
def api_portfolio_sell(pid):
    """Sell stock from portfolio."""
    data = request.json
    shares = int(data.get('shares', 0)) if data.get('shares') else None
    result = db_sell_stock(pid, data['stock_code'], float(data['price']), shares)
    return jsonify(result)


@app.route('/api/portfolio/<pid>/status')
def api_portfolio_status(pid):
    """Get portfolio status and metrics."""
    df = get_cached_data()
    latest_date = df['date'].max()
    prices = df[df['date'] == latest_date].set_index('code')['close'].to_dict()
    status = db_get_portfolio(pid, prices)
    return jsonify(status)


@app.route('/api/trade/quick', methods=['POST'])
def api_quick_trade():
    """Quick buy/sell from trade page, auto-creates portfolio if needed."""
    data = request.json or {}
    action = data.get('action', 'buy')
    stock_code = data.get('stock_code', '')
    stock_name = data.get('stock_name', stock_code)
    price = float(data.get('price', 0))
    shares = int(data.get('shares', 10))
    if not stock_code or price <= 0:
        return jsonify({'success': False, 'error': 'Paramètres invalides'})

    uid = str(session.get('user_id', ''))
    pid = get_user_portfolio(uid)
    if not pid:
        pid = f"pf_{uid}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        db_create_portfolio(pid, 50000, 'moderate', uid)

    if action == 'buy':
        result = db_buy_stock(pid, stock_code, stock_name, price, shares)
    else:
        result = db_sell_stock(pid, stock_code, price, shares)

    if result.get('success'):
        # Get updated portfolio summary
        df = get_cached_data()
        latest_date = df['date'].max()
        prices = df[df['date'] == latest_date].set_index('code')['close'].to_dict()
        pf = db_get_portfolio(pid, prices)
        result['portfolio'] = {
            'pid': pid, 'cash': pf.get('cash', 0),
            'total_value': pf.get('total_value', 0),
            'num_positions': pf.get('num_positions', 0)
        }
    return jsonify(result)


@app.route('/api/portfolio/suggest', methods=['POST'])
def api_portfolio_suggest():
    """Get portfolio suggestions (the '5000 TND' use case). Optimized for speed."""
    try:
        data = request.json or {}
        uid = str(session.get('user_id', ''))
        user = None
        try:
            user = user_manager.get_user(uid)
        except Exception:
            user = None

        capital_default = getattr(user, 'total_capital', 5000) if user else 5000
        profile_default = getattr(user, 'risk_tolerance', 'moderate') if user else 'moderate'
        capital = float(data.get('capital', capital_default))
        profile = data.get('risk_profile', profile_default)
        
        df = get_cached_data()
        engine = DecisionEngine(risk_profile=profile)
        
        # Get top 8 stocks by volume (reduced from 15 for speed)
        top_stocks = df.groupby('code').agg(
            stock=('stock', 'last'),
            volume=('volume', 'sum'),
            last_close=('close', 'last')
        ).nlargest(8, 'volume').reset_index()

        def _analyze_stock(row):
            stock_df = df[df['code'] == row['code']].sort_values('date').tail(500)
            if len(stock_df) < 30:
                return None
            stock_tech = add_technical_indicators(stock_df.tail(100).copy())
            last = stock_tech.iloc[-1]
            forecaster = BVMTForecaster(row['code'])
            forecast = forecaster.forecast(stock_df, horizon=5, fast=True)
            sentiment = get_sentiment_for_stock(row['stock'], row['code'])
            return {
                'stock_name': row['stock'],
                'stock_code': row['code'],
                'current_price': float(last['close']),
                'forecast': forecast,
                'sentiment': sentiment,
                'anomaly': {'has_recent_anomaly': False},
                'technicals': {
                    'rsi': float(last.get('rsi', 50)) if pd.notna(last.get('rsi')) else 50,
                    'macd_hist': float(last.get('macd_hist', 0)) if pd.notna(last.get('macd_hist')) else 0
                }
            }

        # Run all stock analyses in parallel for speed
        futures = {_executor.submit(_analyze_stock, row): row['code'] for _, row in top_stocks.iterrows()}
        stocks_data = []
        for fut in as_completed(futures, timeout=30):
            try:
                result = fut.result()
                if result:
                    stocks_data.append(result)
            except Exception:
                pass
        
        suggestion = engine.generate_portfolio_suggestion(stocks_data, capital=capital)

        # Apply RL corrections if user is logged in
        uid = session.get('user_id')
        if uid and suggestion.get('allocations'):
            rl = PortfolioRL(uid)
            suggestion['allocations'] = rl.adjust_allocations(suggestion['allocations'])
            suggestion['rl_adjusted'] = True

        return jsonify(json_safe({'success': True, **suggestion}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# ══════════════ API — OHLC / TRADE ══════════════

_ohlc_cache = {}
_OHLC_CACHE_TTL = 120  # 2 minutes

@app.route('/api/stock/<stock_code>/ohlc')
def api_stock_ohlc(stock_code):
    """Return OHLC data for candlestick charts."""
    try:
        period = request.args.get('period', '1y')
        cache_key = f"{stock_code}_{period}"
        now = _time.time()
        if cache_key in _ohlc_cache:
            ct, cd = _ohlc_cache[cache_key]
            if now - ct < _OHLC_CACHE_TTL:
                return jsonify(cd)
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').copy()
        if stock_df.empty:
            return jsonify({'success': False, 'error': 'Stock not found'})

        max_date = stock_df['date'].max()
        period_map = {'1m': 30, '3m': 90, '6m': 180, '1y': 365, '2y': 730, 'all': 99999}
        days = period_map.get(period, 365)
        cutoff = max_date - pd.Timedelta(days=days)
        stock_df = stock_df[stock_df['date'] >= cutoff]

        stock_df = add_technical_indicators(stock_df)

        # ── Vectorized OHLC build (no iterrows!) ──
        dates_str = stock_df['date'].dt.strftime('%Y-%m-%d').values
        opens = stock_df['open'].fillna(0).values
        highs = stock_df['high'].fillna(0).values
        lows = stock_df['low'].fillna(0).values
        closes = stock_df['close'].fillna(0).values
        volumes = stock_df['volume'].fillna(0).values

        ohlc = [{'time': dates_str[i], 'open': float(opens[i]), 'high': float(highs[i]),
                 'low': float(lows[i]), 'close': float(closes[i]), 'volume': float(volumes[i])}
                for i in range(len(dates_str))]

        # Indicators — vectorized extraction
        def _extract_indicator(col_name):
            mask = stock_df[col_name].notna() if col_name in stock_df.columns else pd.Series(False, index=stock_df.index)
            vals = stock_df.loc[mask, col_name].values
            dts = stock_df.loc[mask, 'date'].dt.strftime('%Y-%m-%d').values
            return [{'time': dts[i], 'value': round(float(vals[i]), 4)} for i in range(len(vals))]

        sma20 = _extract_indicator('sma_20')
        sma50 = _extract_indicator('sma_50')
        bb_upper = _extract_indicator('bb_upper')
        bb_lower = _extract_indicator('bb_lower')
        rsi_data = _extract_indicator('rsi')

        # MACD needs 3 columns
        macd_mask = stock_df['macd'].notna() if 'macd' in stock_df.columns else pd.Series(False, index=stock_df.index)
        macd_df = stock_df.loc[macd_mask]
        macd_dates = macd_df['date'].dt.strftime('%Y-%m-%d').values
        macd_vals = macd_df['macd'].fillna(0).values
        signal_vals = macd_df['macd_signal'].fillna(0).values
        hist_vals = macd_df['macd_hist'].fillna(0).values
        macd_data = [{'time': macd_dates[i], 'macd': round(float(macd_vals[i]), 4),
                      'signal': round(float(signal_vals[i]), 4), 'hist': round(float(hist_vals[i]), 4)}
                     for i in range(len(macd_dates))]

        stock_name = stock_df['stock'].iloc[0] if 'stock' in stock_df.columns else stock_code
        last = stock_df.iloc[-1]
        prev = stock_df.iloc[-2] if len(stock_df) > 1 else last
        change = safe_float(last['close']) - safe_float(prev['close'])
        change_pct = (change / safe_float(prev['close'], 1)) * 100

        result_data = json_safe({
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'current_price': safe_float(last['close']),
            'change': round(change, 3),
            'change_pct': round(change_pct, 2),
            'ohlc': ohlc,
            'volume': [{'time': o['time'], 'value': o['volume'],
                        'color': 'rgba(63,185,80,0.5)' if o['close'] >= o['open'] else 'rgba(248,81,73,0.5)'}
                       for o in ohlc],
            'indicators': {
                'sma20': sma20, 'sma50': sma50,
                'bb_upper': bb_upper, 'bb_lower': bb_lower,
                'rsi': rsi_data, 'macd': macd_data
            }
        })
        _ohlc_cache[cache_key] = (_time.time(), result_data)
        return jsonify(result_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


_sarima_cache = {}
_SARIMA_CACHE_TTL = 1800  # 30 minutes

@app.route('/api/stock/<stock_code>/sarima')
def api_stock_sarima(stock_code):
    """Return detailed SARIMA model statistics for a stock."""
    try:
        now = _time.time()
        if stock_code in _sarima_cache:
            ct, cd = _sarima_cache[stock_code]
            if now - ct < _SARIMA_CACHE_TTL:
                return jsonify(cd)
        df = get_cached_data()
        stock_df = df[df['code'] == stock_code].sort_values('date').copy()
        if stock_df.empty:
            return jsonify({'success': False, 'error': 'Stock not found'})

        close_series = stock_df['close'].dropna()
        volume_series = stock_df['volume'].dropna()

        forecaster = BVMTForecaster(stock_code)

        include_xgb = request.args.get('xgb', '0').strip().lower() in ('1', 'true', 'yes')

        # Stationarity
        stationarity = forecaster.check_stationarity(close_series, 'close_price')
        vol_stationarity = forecaster.check_stationarity(volume_series, 'volume')

        # EMA model stats (replaces SARIMA)
        ema_result = forecaster.fit_ema_forecast(close_series, 5)
        ema_stats = {}
        if ema_result:
            ema_stats = {
                'method': 'EMA_extrapolation',
                'ema10': ema_result.get('ema10'),
                'ema20': ema_result.get('ema20'),
                'trend_per_day': ema_result.get('trend_per_day'),
                'forecast': ema_result.get('forecast'),
            }

        # Linear regression stats (replaces ETS)
        lr_result = forecaster.fit_linear_forecast(close_series, 5)
        lr_stats = {}
        if lr_result:
            lr_stats = {
                'method': 'weighted_linear_regression',
                'slope_per_day': lr_result.get('slope_per_day'),
                'r_squared': lr_result.get('r_squared'),
                'forecast': lr_result.get('forecast'),
            }

        # XGBoost metrics (opt-in only — can be heavy)
        xgb_stats = {}
        if include_xgb:
            try:
                _, xgb_metrics = forecaster.fit_xgboost(stock_df)
                xgb_stats = xgb_metrics if xgb_metrics else {}
            except Exception:
                xgb_stats = {}

        # Backtest evaluation
        backtest = forecaster.evaluate_backtest(stock_df, horizon=5)

        # Variance stats
        cv = float(close_series.std() / close_series.mean())
        skewness = float(close_series.skew())
        kurtosis = float(close_series.kurtosis())

        # Recent price stats
        last_20 = close_series.tail(20)
        last_60 = close_series.tail(60)

        sarima_result = json_safe({
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_df['stock'].iloc[0] if 'stock' in stock_df.columns else stock_code,
            'data_points': len(close_series),
            'date_range': [str(stock_df['date'].min().date()), str(stock_df['date'].max().date())],
            'price_stats': {
                'current': round(float(close_series.iloc[-1]), 3),
                'mean': round(float(close_series.mean()), 3),
                'std': round(float(close_series.std()), 3),
                'min': round(float(close_series.min()), 3),
                'max': round(float(close_series.max()), 3),
                'cv': round(cv, 4),
                'skewness': round(skewness, 4),
                'kurtosis': round(kurtosis, 4),
                'ma20': round(float(last_20.mean()), 3),
                'ma60': round(float(last_60.mean()), 3),
                'volatility_20d': round(float(last_20.pct_change().std() * 100), 3),
            },
            'stationarity': stationarity,
            'volume_stationarity': vol_stationarity,
            'sarima': ema_stats,
            'ets': lr_stats,
            'xgboost': xgb_stats,
            'backtest': backtest,
        })
        _sarima_cache[stock_code] = (_time.time(), sarima_result)
        return jsonify(sarima_result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/portfolio/sarima_dashboard', methods=['POST'])
def api_portfolio_sarima_dashboard():
    """SARIMA stats for all stocks in a portfolio or suggested stocks."""
    try:
        data = request.json or {}
        stock_codes = data.get('stock_codes', [])
        if not stock_codes:
            df = get_cached_data()
            latest = df['date'].max()
            today = df[df['date'] == latest]
            top5 = today.nlargest(3, 'volume')['code'].tolist()
            stock_codes = top5

        df = get_cached_data()
        results = []
        for code in stock_codes[:5]:  # Limit to 5 for speed
            stock_df = df[df['code'] == code].sort_values('date').tail(500)
            if stock_df.empty or len(stock_df) < 30:
                continue
            close = stock_df['close'].dropna()
            forecaster = BVMTForecaster(code)
            forecast = forecaster.forecast(stock_df, horizon=5, fast=True)

            entry = {
                'stock_code': code,
                'stock_name': stock_df['stock'].iloc[0] if 'stock' in stock_df.columns else code,
                'current_price': round(float(close.iloc[-1]), 3),
                'forecast': forecast.get('ensemble', {}).get('forecast', []),
                'forecast_dates': forecast.get('forecast_dates', []),
                'lower_ci': forecast.get('ensemble', {}).get('lower_ci', []),
                'upper_ci': forecast.get('ensemble', {}).get('upper_ci', []),
                'num_models': forecast.get('ensemble', {}).get('num_models', 0),
            }
            # Stationarity (ADF test)
            stat = forecast.get('stationarity', {})
            entry['stationary'] = stat.get('stationary', False)
            entry['adf_statistic'] = stat.get('adf_statistic', None)
            entry['adf_pvalue'] = stat.get('adf_pvalue', None)
            entry['adf_recommendation'] = stat.get('recommendation', '')

            # AIC/BIC from model selection
            model_sel = forecast.get('model_selection', {})
            sarima_res = forecast.get('sarima', {})
            ets_res = forecast.get('ets', {})
            entry['aic'] = model_sel.get('best_aic') or sarima_res.get('aic')
            entry['bic'] = model_sel.get('best_bic') or sarima_res.get('bic')
            entry['aic_details'] = model_sel.get('all_aic', {})
            entry['bic_details'] = model_sel.get('all_bic', {})

            # Backtest metrics (RMSE, MAE, DA)
            bt = forecast.get('backtest_metrics', {})
            entry['rmse'] = bt.get('rmse')
            entry['mae'] = bt.get('mae')
            entry['directional_accuracy'] = bt.get('directional_accuracy')

            results.append(entry)

        return jsonify(json_safe({'success': True, 'stocks': results}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# ══════════════ API — CHAT ══════════════

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({'success': False, 'error': 'Empty message'})
    try:
        df = get_cached_data()
        latest_date = df['date'].max()
        prev_date = df[df['date'] < latest_date]['date'].max()
        today = df[df['date'] == latest_date]
        yesterday = df[df['date'] == prev_date]
        merged = today.merge(yesterday[['code', 'close']], on='code', suffixes=('', '_prev'))
        merged['chg'] = ((merged['close'] - merged['close_prev']) / merged['close_prev'] * 100).round(2)
        gainers = merged.nlargest(5, 'chg')[['stock', 'close', 'chg']].to_dict('records')
        losers = merged.nsmallest(5, 'chg')[['stock', 'close', 'chg']].to_dict('records')
        total_vol = int(today['volume'].sum())
        advancing = int((merged['chg'] > 0).sum())
        declining = int((merged['chg'] < 0).sum())
        context = {
            'date': str(latest_date.date()),
            'num_stocks': int(today['code'].nunique()),
            'market': 'BVMT',
            'total_volume': total_vol,
            'advancing': advancing, 'declining': declining,
            'top_gainers': gainers[:3],
            'top_losers': losers[:3]
        }
    except Exception:
        context = {}

    # Add user profile to context if logged in
    try:
        uid = session.get('user_id')
        if uid:
            user = user_manager.get_user(uid)
            if user:
                context['user_profile'] = user.get_profile_summary()
                context['user_name'] = user.display_name

            # Add portfolio data for AI suggestions
            pid = get_user_portfolio(uid)
            if pid:
                prices = {}
                try:
                    latest_date_pf = df['date'].max() if 'df' in dir() else None
                    if latest_date_pf is not None:
                        prices = df[df['date'] == latest_date_pf].set_index('code')['close'].to_dict()
                except Exception:
                    pass
                pf = db_get_portfolio(pid, prices)
                if pf.get('success'):
                    context['portfolio'] = {
                        'capital_liquide': pf.get('cash', 0),
                        'valeur_titres': pf.get('invested', 0),
                        'valeur_totale': pf.get('total_value', 0),
                        'roi_pct': pf.get('roi_pct', 0),
                        'capital_initial': pf.get('initial_capital', 0),
                        'risk_profile': pf.get('risk_profile', 'moderate'),
                        'nb_positions': pf.get('num_positions', 0),
                        'positions': [
                            {'titre': p.get('name', p.get('code', '')),
                             'code': p.get('code', ''),
                             'quantite': p.get('shares', 0),
                             'pru': p.get('avg_price', 0),
                             'prix_actuel': p.get('current_price', 0),
                             'pnl_pct': p.get('pnl_pct', 0)}
                            for p in (pf.get('positions') or [])[:10]
                        ]
                    }
    except Exception:
        pass

    try:
        response = chat_agent.chat(message, context=context)
    except Exception as e:
        response = f'Erreur: {str(e)}'
    return jsonify({'success': True, 'response': response})


@app.route('/api/chat/clear', methods=['POST'])
def api_chat_clear():
    chat_agent.clear_history()
    return jsonify({'success': True})


# ══════════════ API — AUTH & USER ══════════════

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.json or {}
    result = user_manager.register(
        username=data.get('username', ''),
        password=data.get('password', ''),
        display_name=data.get('display_name', '')
    )
    return jsonify(result)


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.json or {}
    result = user_manager.login(data.get('username', ''), data.get('password', ''))
    if result.get('success'):
        session['user_id'] = result['user_id']
        session['username'] = result['username']
    return jsonify(result)


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/api/auth/me')
def api_me():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'success': False, 'logged_in': False})
    user = user_manager.get_user(uid)
    if not user:
        return jsonify({'success': False, 'logged_in': False})
    return jsonify({'success': True, 'logged_in': True, 'user': user.to_dict()})


@app.route('/api/user/profile', methods=['POST'])
def api_update_profile():
    uid = session.get('user_id')
    if not uid:
        return jsonify({'success': False, 'error': 'Non connecté'})
    updates = request.json or {}
    return jsonify(user_manager.update_profile(uid, updates))


@app.route('/api/user/onboarding', methods=['POST'])
def api_onboarding():
    """Process chatbot onboarding answers to build user profile."""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'success': False, 'error': 'Non connecté'})
    data = request.json or {}

    updates = {}
    if 'experience' in data:
        updates['investment_experience'] = data['experience']
    if 'risk' in data:
        updates['risk_tolerance'] = data['risk']
    if 'horizon' in data:
        updates['investment_horizon'] = data['horizon']
    if 'budget' in data:
        updates['monthly_budget'] = float(data.get('budget', 0))
    if 'capital' in data:
        updates['total_capital'] = float(data.get('capital', 5000))
    if 'sectors' in data:
        updates['preferred_sectors'] = data['sectors'] if isinstance(data['sectors'], list) else [data['sectors']]
    if 'goals' in data:
        updates['investment_goals'] = data['goals']
    if 'age' in data:
        updates['age_range'] = data['age']
    # Any successful onboarding submission completes the profile.
    updates['profile_completed'] = True

    result = user_manager.update_profile(uid, updates)

    # ── Auto-create portfolio if user doesn't have one yet ──
    if result.get('success'):
        existing_pid = get_user_portfolio(uid)
        if not existing_pid:
            capital = float(data.get('capital', 5000))
            risk = data.get('risk', 'moderate')
            pid = f"pf_{uid}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            pf_result = db_create_portfolio(pid, capital, risk, uid)
            if pf_result.get('success'):
                result['portfolio_created'] = True
                result['portfolio_id'] = pid
                result['initial_capital'] = capital
                result['risk_profile'] = risk

    return jsonify(result)


# ══════════════ API — RL FEEDBACK ══════════════

@app.route('/api/rl/feedback', methods=['POST'])
def api_rl_feedback():
    """Record user feedback for RL learning."""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'success': False, 'error': 'Non connecté'})
    data = request.json or {}
    rl = PortfolioRL(uid)
    result = rl.record_feedback(
        stock_code=data.get('stock_code', ''),
        stock_name=data.get('stock_name', ''),
        action=data.get('action', 'buy'),
        liked=data.get('liked', True),
        profit_pct=float(data.get('profit_pct', 0)),
        sector=data.get('sector', ''),
        reason=data.get('reason', '')
    )
    return jsonify(result)


@app.route('/api/rl/summary')
def api_rl_summary():
    """Get RL model summary for current user."""
    uid = session.get('user_id')
    if not uid:
        return jsonify({'success': False, 'error': 'Non connecté'})
    rl = PortfolioRL(uid)
    return jsonify({'success': True, **rl.get_model_summary()})


# ══════════════ API — REAL-TIME DATA ══════════════

@app.route('/api/realtime/status')
def api_realtime_status():
    """Get real-time scraper status."""
    return jsonify({'success': True, **realtime_scraper.get_status()})


@app.route('/api/realtime/latest')
def api_realtime_latest():
    """Get latest prices for all stocks."""
    data = realtime_scraper.get_latest_all()
    return jsonify({'success': True, 'stocks': data, 'count': len(data)})


@app.route('/api/realtime/quote/<ticker>')
def api_realtime_quote(ticker):
    """Get latest quote for a stock."""
    data = realtime_scraper.get_latest(ticker.upper())
    if data:
        return jsonify({'success': True, 'quote': data})
    return jsonify({'success': False, 'error': f'Ticker {ticker} non trouvé'})


@app.route('/api/realtime/candles/<ticker>')
def api_realtime_candles(ticker):
    """Get OHLCV candles for a stock."""
    tf = request.args.get('timeframe', '5m')
    limit = int(request.args.get('limit', 100))
    candles = realtime_scraper.get_candles(ticker.upper(), tf, limit)
    return jsonify({'success': True, 'ticker': ticker.upper(), 'timeframe': tf,
                    'candles': candles, 'count': len(candles)})


@app.route('/api/realtime/search')
def api_realtime_search():
    """Search for a ticker."""
    q = request.args.get('q', '')
    results = realtime_scraper.search_ticker(q)
    return jsonify({'success': True, 'results': results})


@app.route('/api/realtime/files')
def api_realtime_files():
    """List persisted scraper data files."""
    files = realtime_scraper.get_persisted_files()
    return jsonify({'success': True, 'files': files})


@app.route('/api/realtime/dashboard')
def api_realtime_dashboard():
    """Get comprehensive real-time data for dashboard display."""
    latest = realtime_scraper.get_latest_all()
    status = realtime_scraper.get_status()
    files = realtime_scraper.get_persisted_files()

    # Sort by variation for top movers
    stocks_list = []
    for ticker, data in latest.items():
        stocks_list.append({
            'ticker': ticker,
            'name': data.get('name', ticker),
            'price': data.get('price', 0),
            'open': data.get('open', 0),
            'high': data.get('high', 0),
            'low': data.get('low', 0),
            'volume': data.get('volume', 0),
            'var_pct': data.get('var_pct', 0),
            'time': data.get('time', ''),
        })

    stocks_list.sort(key=lambda x: abs(x.get('var_pct', 0)), reverse=True)

    return jsonify({
        'success': True,
        'status': status,
        'stocks': stocks_list,
        'count': len(stocks_list),
        'files': files,
    })


# ══════════════ RUN ══════════════

if __name__ == '__main__':
    print("=" * 60, flush=True)
    print("  Tradeili — Trading Assistant by the overfitters", flush=True)
    print("  http://localhost:5000", flush=True)
    print("  Data ready — server starting...", flush=True)
    print("=" * 60, flush=True)
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False, threaded=True)
