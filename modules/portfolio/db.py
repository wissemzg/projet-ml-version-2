"""
SQLite-backed Portfolio Storage for BVMT Trading Assistant.
Fast persistent storage for portfolios, positions, and trades.
"""
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "portfolio.db"


def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Create tables if not exist."""
    conn = _get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS portfolios (
        pid TEXT PRIMARY KEY,
        user_id TEXT DEFAULT '',
        initial_capital REAL DEFAULT 5000,
        cash REAL DEFAULT 5000,
        risk_profile TEXT DEFAULT 'moderate',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid TEXT NOT NULL,
        stock_code TEXT NOT NULL,
        stock_name TEXT DEFAULT '',
        shares INTEGER DEFAULT 0,
        avg_price REAL DEFAULT 0,
        date_bought TEXT DEFAULT (date('now')),
        FOREIGN KEY (pid) REFERENCES portfolios(pid),
        UNIQUE(pid, stock_code)
    );
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pid TEXT NOT NULL,
        type TEXT NOT NULL,
        stock_code TEXT NOT NULL,
        stock_name TEXT DEFAULT '',
        shares INTEGER DEFAULT 0,
        price REAL DEFAULT 0,
        total REAL DEFAULT 0,
        profit REAL DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (pid) REFERENCES portfolios(pid)
    );
    CREATE INDEX IF NOT EXISTS idx_positions_pid ON positions(pid);
    CREATE INDEX IF NOT EXISTS idx_trades_pid ON trades(pid);
    """)
    conn.commit()
    conn.close()


# Initialize on import
init_db()


def create_portfolio(pid: str, capital: float = 5000, risk_profile: str = 'moderate', user_id: str = '') -> dict:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO portfolios (pid, user_id, initial_capital, cash, risk_profile) VALUES (?,?,?,?,?)",
            (pid, user_id, capital, capital, risk_profile)
        )
        conn.commit()
        return {"success": True, "portfolio_id": pid, "capital": capital, "risk_profile": risk_profile}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Portfolio already exists"}
    finally:
        conn.close()


def buy_stock(pid: str, stock_code: str, stock_name: str, price: float, shares: int) -> dict:
    conn = _get_conn()
    try:
        row = conn.execute("SELECT cash FROM portfolios WHERE pid=?", (pid,)).fetchone()
        if not row:
            return {"success": False, "error": "Portfolio not found"}

        cash = row['cash']
        cost = shares * price
        if cost > cash or shares <= 0:
            return {"success": False, "error": f"Fonds insuffisants ({cash:.2f} TND disponible, coût: {cost:.2f} TND)"}

        # Update cash
        conn.execute("UPDATE portfolios SET cash=cash-? WHERE pid=?", (cost, pid))

        # Upsert position
        existing = conn.execute(
            "SELECT shares, avg_price FROM positions WHERE pid=? AND stock_code=?",
            (pid, stock_code)
        ).fetchone()

        if existing:
            old_shares = existing['shares']
            old_avg = existing['avg_price']
            new_shares = old_shares + shares
            new_avg = (old_avg * old_shares + price * shares) / new_shares
            conn.execute(
                "UPDATE positions SET shares=?, avg_price=?, stock_name=? WHERE pid=? AND stock_code=?",
                (new_shares, new_avg, stock_name, pid, stock_code)
            )
        else:
            conn.execute(
                "INSERT INTO positions (pid, stock_code, stock_name, shares, avg_price) VALUES (?,?,?,?,?)",
                (pid, stock_code, stock_name, shares, price)
            )

        # Record trade
        conn.execute(
            "INSERT INTO trades (pid, type, stock_code, stock_name, shares, price, total) VALUES (?,?,?,?,?,?,?)",
            (pid, 'BUY', stock_code, stock_name, shares, price, cost)
        )
        conn.commit()
        return {"success": True, "type": "BUY", "stock": stock_code, "name": stock_name,
                "shares": shares, "price": price, "cost": round(cost, 2)}
    finally:
        conn.close()


def sell_stock(pid: str, stock_code: str, price: float, shares: int | None = None) -> dict:
    conn = _get_conn()
    try:
        pos = conn.execute(
            "SELECT shares, avg_price, stock_name FROM positions WHERE pid=? AND stock_code=?",
            (pid, stock_code)
        ).fetchone()
        if not pos:
            return {"success": False, "error": "Aucune position sur ce titre"}

        max_shares = pos['shares']
        if shares is None:
            shares = max_shares
        shares = int(min(shares, max_shares))

        revenue = shares * price
        profit = (price - pos['avg_price']) * shares

        conn.execute("UPDATE portfolios SET cash=cash+? WHERE pid=?", (revenue, pid))

        remaining = max_shares - shares
        if remaining <= 0:
            conn.execute("DELETE FROM positions WHERE pid=? AND stock_code=?", (pid, stock_code))
        else:
            conn.execute("UPDATE positions SET shares=? WHERE pid=? AND stock_code=?",
                         (remaining, pid, stock_code))

        conn.execute(
            "INSERT INTO trades (pid, type, stock_code, stock_name, shares, price, total, profit) VALUES (?,?,?,?,?,?,?,?)",
            (pid, 'SELL', stock_code, pos['stock_name'], shares, price, revenue, profit)
        )
        conn.commit()
        return {"success": True, "type": "SELL", "stock": stock_code, "name": pos['stock_name'],
                "shares": shares, "price": price, "revenue": round(revenue, 2), "profit": round(profit, 2)}
    finally:
        conn.close()


def get_portfolio(pid: str, current_prices: dict | None = None) -> dict:
    conn = _get_conn()
    try:
        pf = conn.execute("SELECT * FROM portfolios WHERE pid=?", (pid,)).fetchone()
        if not pf:
            return {"success": False, "error": "Portfolio not found"}

        positions = conn.execute(
            "SELECT * FROM positions WHERE pid=? AND shares>0", (pid,)
        ).fetchall()

        trades = conn.execute(
            "SELECT * FROM trades WHERE pid=? ORDER BY created_at DESC LIMIT 50", (pid,)
        ).fetchall()

        current_prices = current_prices or {}
        positions_value = 0
        pos_list = []
        for p in positions:
            cp = current_prices.get(p['stock_code'], p['avg_price'])
            val = p['shares'] * cp
            pnl = (cp - p['avg_price']) * p['shares']
            pnl_pct = ((cp / p['avg_price']) - 1) * 100 if p['avg_price'] > 0 else 0
            positions_value += val
            pos_list.append({
                'code': p['stock_code'], 'name': p['stock_name'],
                'shares': p['shares'], 'avg_price': round(p['avg_price'], 3),
                'current_price': round(cp, 3), 'value': round(val, 2),
                'pnl': round(pnl, 2), 'pnl_pct': round(pnl_pct, 2)
            })

        total_value = pf['cash'] + positions_value
        roi = ((total_value / pf['initial_capital']) - 1) * 100

        for p in pos_list:
            p['allocation_pct'] = round((p['value'] / total_value) * 100, 1) if total_value > 0 else 0

        return {
            "success": True,
            "portfolio_id": pid,
            "total_value": round(total_value, 2),
            "cash": round(pf['cash'], 2),
            "available_cash": round(pf['cash'], 2),
            "invested": round(positions_value, 2),
            "roi_pct": round(roi, 2),
            "initial_capital": pf['initial_capital'],
            "num_positions": len(pos_list),
            "positions": sorted(pos_list, key=lambda x: x['value'], reverse=True),
            "holdings": pos_list,
            "risk_profile": pf['risk_profile'],
            "trades": [dict(t) for t in trades]
        }
    finally:
        conn.close()


def get_user_portfolio(user_id: str) -> str | None:
    """Get or create default portfolio for a user."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT pid FROM portfolios WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
                           (user_id,)).fetchone()
        if row:
            return row['pid']
        return None
    finally:
        conn.close()


def list_portfolios(user_id: str = '') -> list:
    conn = _get_conn()
    try:
        if user_id:
            rows = conn.execute("SELECT pid, initial_capital, cash, risk_profile, created_at FROM portfolios WHERE user_id=?",
                                (user_id,)).fetchall()
        else:
            rows = conn.execute("SELECT pid, initial_capital, cash, risk_profile, created_at FROM portfolios").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
