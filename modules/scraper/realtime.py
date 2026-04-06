"""
Real-Time Market Data Scraper for BVMT.
Scrapes live stock prices from ilboursa.com and bvmt.com.tn.
Provides multi-timeframe data (1min, 5min, 15min, 1h) via in-memory cache.
Persists snapshots to data/scraper/ for visibility and data recovery.
"""
import re
import time
import json
import threading
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional
from pathlib import Path
from bs4 import BeautifulSoup


class RealtimeScraper:
    """
    Scrapes real-time quotes from Tunisian stock market sources.
    Stores tick data in memory for multi-timeframe candle generation.
    Persists snapshots to data/scraper/ after each scrape cycle.
    """

    ILBOURSA_AZ = "https://www.ilboursa.com/marches/aaz"
    ILBOURSA_STOCK = "https://www.ilboursa.com/marches/cotation_{}"
    BVMT_TICKER = "https://www.bvmt.com.tn/public/BvmtTicker/index.html"

    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # In-memory tick storage: {stock_code: [{time, price, volume, var_pct}]}
        self._ticks: Dict[str, list] = defaultdict(list)
        # Latest snapshot: {stock_code: {price, var, volume, ...}}
        self._latest: Dict[str, dict] = {}
        # OHLCV candles: {stock_code: {'1m': [...], '5m': [...], ...}}
        self._candles: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        # Last scrape time
        self._last_scrape = None
        # Scrape errors
        self._errors: list = []
        # Scrape counter for persistence
        self._scrape_count = 0

        # Code-to-ticker mapping (built from ilboursa)
        self._ticker_map: Dict[str, str] = {}  # internal_code -> ticker

        # Persistence directory
        self._data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "scraper"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_data()

    def start(self):
        """Start background scraping."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self._thread.start()
        print(f"[SCRAPER] Started real-time scraping (interval={self.interval}s)", flush=True)

    def stop(self):
        """Stop background scraping."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[SCRAPER] Stopped", flush=True)

    def _scrape_loop(self):
        while self._running:
            try:
                self._scrape_all()
            except Exception as e:
                self._errors.append({'time': datetime.now().isoformat(), 'error': str(e)})
                if len(self._errors) > 50:
                    self._errors = self._errors[-50:]
            time.sleep(self.interval)

    def _scrape_all(self):
        """Scrape all stocks from ilboursa A-Z page."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(self.ILBOURSA_AZ, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')

            now = datetime.now()
            self._last_scrape = now.isoformat()

            # Find the stock table
            table = soup.find('table', {'class': 'table'}) or soup.find('table')
            if not table:
                # Try parsing ticker-style data
                self._scrape_bvmt_ticker()
                return

            rows = table.find_all('tr')[1:]  # Skip header
            count = 0
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 5:
                    continue

                try:
                    # Parse stock name and link
                    link = cells[0].find('a')
                    name = link.get_text(strip=True) if link else cells[0].get_text(strip=True)
                    href = link.get('href', '') if link else ''

                    # Extract ticker from URL (e.g., /marches/cotation_BIAT -> BIAT)
                    ticker = ''
                    if 'cotation_' in href:
                        ticker = href.split('cotation_')[-1].strip('/')
                    if not ticker:
                        ticker = name.replace(' ', '_')[:10]

                    # Parse numeric values
                    values = []
                    for c in cells[1:]:
                        txt = c.get_text(strip=True).replace('\xa0', '').replace(' ', '')
                        txt = txt.replace(',', '.')
                        try:
                            values.append(float(txt))
                        except ValueError:
                            values.append(0.0)

                    # Typical columns: Ouverture, Plus Haut, Plus Bas, Dernier, Volume, Variation%
                    if len(values) >= 5:
                        tick = {
                            'time': now.isoformat(),
                            'timestamp': now.timestamp(),
                            'name': name,
                            'ticker': ticker,
                            'open': values[0] if values[0] > 0 else values[3],
                            'high': values[1] if values[1] > 0 else values[3],
                            'low': values[2] if values[2] > 0 else values[3],
                            'price': values[3],
                            'volume': int(values[4]) if len(values) > 4 else 0,
                            'var_pct': values[-1] if len(values) > 5 else 0.0,
                        }

                        with self._lock:
                            self._latest[ticker] = tick
                            self._ticks[ticker].append(tick)
                            # Keep last 1000 ticks per stock
                            if len(self._ticks[ticker]) > 1000:
                                self._ticks[ticker] = self._ticks[ticker][-1000:]
                            self._update_candles(ticker, tick)
                        count += 1

                except Exception:
                    continue

            if count > 0:
                print(f"[SCRAPER] {now.strftime('%H:%M:%S')} — {count} stocks updated", flush=True)
                self._scrape_count += 1
                # Persist every 5 scrapes (every ~5 minutes)
                if self._scrape_count % 5 == 0:
                    self._persist_data()

        except requests.RequestException as e:
            # Fallback to BVMT ticker
            self._scrape_bvmt_ticker()

    def _scrape_bvmt_ticker(self):
        """Fallback: scrape BVMT ticker page."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(self.BVMT_TICKER, headers=headers, timeout=10)
            resp.raise_for_status()

            now = datetime.now()
            # Parse "TICKER PRICE (VAR%)" patterns
            pattern = r'([A-Z][A-Z0-9\s]{1,20}?)\s+([\d,.]+)\s+\(([-+]?[\d,.]+)\)'
            matches = re.findall(pattern, resp.text)

            for name, price_str, var_str in matches:
                name = name.strip()
                ticker = name.replace(' ', '_')[:10]
                try:
                    price = float(price_str.replace(',', '.'))
                    var_pct = float(var_str.replace(',', '.'))
                except ValueError:
                    continue

                tick = {
                    'time': now.isoformat(),
                    'timestamp': now.timestamp(),
                    'name': name, 'ticker': ticker,
                    'open': price, 'high': price, 'low': price,
                    'price': price, 'volume': 0,
                    'var_pct': var_pct,
                }

                with self._lock:
                    self._latest[ticker] = tick
                    self._ticks[ticker].append(tick)
                    if len(self._ticks[ticker]) > 1000:
                        self._ticks[ticker] = self._ticks[ticker][-1000:]
                    self._update_candles(ticker, tick)

        except Exception as e:
            self._errors.append({'time': datetime.now().isoformat(), 'error': f'BVMT fallback: {e}'})

    def _update_candles(self, ticker: str, tick: dict):
        """Update OHLCV candles for all timeframes."""
        ts = tick['timestamp']
        price = tick['price']
        volume = tick.get('volume', 0)

        for tf, seconds in [('1m', 60), ('5m', 300), ('15m', 900), ('1h', 3600)]:
            candles = self._candles[ticker][tf]
            candle_start = int(ts // seconds) * seconds

            if candles and candles[-1]['t_start'] == candle_start:
                # Update existing candle
                c = candles[-1]
                c['high'] = max(c['high'], price)
                c['low'] = min(c['low'], price)
                c['close'] = price
                c['volume'] += volume
                c['ticks'] += 1
            else:
                # New candle
                candles.append({
                    't_start': candle_start,
                    'time': datetime.fromtimestamp(candle_start).isoformat(),
                    'open': price, 'high': price, 'low': price,
                    'close': price, 'volume': volume, 'ticks': 1,
                })
                # Keep last 500 candles per timeframe
                if len(candles) > 500:
                    self._candles[ticker][tf] = candles[-500:]

    # ── Persistence ──

    def _persist_data(self):
        """Save latest snapshot and recent ticks to disk for visibility."""
        try:
            with self._lock:
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'stocks_count': len(self._latest),
                    'latest': dict(self._latest),
                }
            # Save latest snapshot
            snapshot_file = self._data_dir / 'latest_snapshot.json'
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)

            # Save daily log (append mode)
            daily_file = self._data_dir / f"ticks_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(daily_file, 'a', encoding='utf-8') as f:
                for ticker, data in snapshot.get('latest', {}).items():
                    entry = {'ticker': ticker, **data}
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        except Exception as e:
            self._errors.append({'time': datetime.now().isoformat(), 'error': f'Persist: {e}'})

    def _load_persisted_data(self):
        """Load latest snapshot from disk on startup."""
        try:
            snapshot_file = self._data_dir / 'latest_snapshot.json'
            if snapshot_file.exists():
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('latest'):
                    self._latest = data['latest']
                    print(f"[SCRAPER] Loaded {len(self._latest)} stocks from persisted snapshot", flush=True)
        except Exception as e:
            print(f"[SCRAPER] Could not load persisted data: {e}", flush=True)

    def get_persisted_files(self) -> list:
        """List all persisted data files for visibility."""
        files = []
        for f in sorted(self._data_dir.glob('*.json*')):
            stat = f.stat()
            files.append({
                'name': f.name,
                'size_kb': round(stat.st_size / 1024, 1),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return files

    # ── Public API ──

    def get_latest_all(self) -> Dict[str, dict]:
        """Get latest price for all stocks."""
        with self._lock:
            return dict(self._latest)

    def get_latest(self, ticker: str) -> Optional[dict]:
        """Get latest price for one stock."""
        with self._lock:
            return self._latest.get(ticker)

    def get_candles(self, ticker: str, timeframe: str = '5m', limit: int = 100) -> list:
        """Get OHLCV candles for a stock."""
        with self._lock:
            candles = self._candles.get(ticker, {}).get(timeframe, [])
            return candles[-limit:]

    def get_ticks(self, ticker: str, limit: int = 100) -> list:
        """Get raw tick data."""
        with self._lock:
            return self._ticks.get(ticker, [])[-limit:]

    def get_status(self) -> dict:
        """Get scraper status."""
        with self._lock:
            return {
                'running': self._running,
                'last_scrape': self._last_scrape,
                'stocks_tracked': len(self._latest),
                'tickers': sorted(self._latest.keys()),
                'interval_seconds': self.interval,
                'errors_recent': self._errors[-5:],
            }

    def get_available_timeframes(self) -> list:
        return ['1m', '5m', '15m', '1h']

    def search_ticker(self, query: str) -> list:
        """Search tickers by name or code."""
        q = query.lower()
        results = []
        with self._lock:
            for ticker, data in self._latest.items():
                if q in ticker.lower() or q in data.get('name', '').lower():
                    results.append({'ticker': ticker, 'name': data.get('name', ''),
                                     'price': data.get('price', 0), 'var_pct': data.get('var_pct', 0)})
        return results[:20]
