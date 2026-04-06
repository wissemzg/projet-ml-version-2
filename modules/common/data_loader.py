"""
Common data loader for BVMT historical quotation data.
Handles both histo_cotation and web_histo_cotation CSV files.
"""
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

COLUMN_MAP = {
    'SEANCE': 'date',
    'GROUPE': 'group',
    'CODE': 'code',
    'VALEUR': 'stock',
    'OUVERTURE': 'open',
    'CLOTURE': 'close',
    'PLUS_BAS': 'low',
    'PLUS_HAUT': 'high',
    'QUANTITE_NEGOCIEE': 'volume',
    'NB_TRANSACTION': 'transactions',
    'CAPITAUX': 'capital'
}


def load_single_csv(filepath: str) -> pd.DataFrame:
    """Load a single BVMT CSV file."""
    df = pd.read_csv(filepath, sep=';', encoding='utf-8', skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=COLUMN_MAP, inplace=True)
    
    # Clean string columns
    for col in ['stock', 'code', 'group']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'].str.strip(), format='%d/%m/%Y', errors='coerce')
    
    # Parse numeric columns
    for col in ['open', 'close', 'low', 'high', 'volume', 'transactions', 'capital']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
    
    df.dropna(subset=['date'], inplace=True)
    return df


def load_single_txt(filepath: str) -> pd.DataFrame:
    """Load a fixed-width BVMT .txt file (2016-2021 format)."""
    colspecs = [
        (0, 10),    # SEANCE
        (11, 18),   # GROUPE
        (19, 25),   # CODE
        (26, 44),   # VALEUR
        (45, 56),   # OUVERTURE
        (57, 68),   # CLOTURE
        (69, 80),   # PLUS_BAS
        (81, 92),   # PLUS_HAUT
        (93, 111),  # QUANTITE_NEGOCIEE
        (112, 127), # NB_TRANSACTION
        (128, 145), # CAPITAUX
    ]
    col_names = ['SEANCE', 'GROUPE', 'CODE', 'VALEUR', 'OUVERTURE', 'CLOTURE',
                 'PLUS_BAS', 'PLUS_HAUT', 'QUANTITE_NEGOCIEE', 'NB_TRANSACTION', 'CAPITAUX']
    
    df = pd.read_fwf(filepath, colspecs=colspecs, names=col_names,
                     encoding='latin-1', skiprows=2)
    df.rename(columns=COLUMN_MAP, inplace=True)
    
    for col in ['stock', 'code', 'group']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='%d/%m/%Y', errors='coerce')
    
    for col in ['open', 'close', 'low', 'high', 'volume', 'transactions', 'capital']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.replace(',', '.'),
                errors='coerce'
            )
    
    df.dropna(subset=['date'], inplace=True)
    return df


def load_all_data() -> pd.DataFrame:
    """Load and concatenate all BVMT data files (.csv and .txt)."""
    csv_files = sorted(glob.glob(str(DATA_DIR / "histo_cotation_*.csv"))) + \
                sorted(glob.glob(str(DATA_DIR / "web_histo_cotation_*.csv")))
    txt_files = sorted(glob.glob(str(DATA_DIR / "histo_cotation_*.txt")))
    
    if not csv_files and not txt_files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")
    
    dfs = []
    for f in csv_files:
        try:
            df = load_single_csv(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    for f in txt_files:
        try:
            df = load_single_txt(f)
            df['source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.drop_duplicates(subset=['date', 'code'], keep='last', inplace=True)
    combined.sort_values(['code', 'date'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def get_stock_data(stock_name: str = None, stock_code: str = None) -> pd.DataFrame:
    """Get data for a specific stock by name or code."""
    df = load_all_data()
    if stock_code:
        df = df[df['code'] == stock_code]
    elif stock_name:
        df = df[df['stock'].str.contains(stock_name, case=False, na=False)]
    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_stock_list() -> pd.DataFrame:
    """Get list of all unique stocks."""
    df = load_all_data()
    stocks = df.groupby(['code', 'stock']).agg(
        first_date=('date', 'min'),
        last_date=('date', 'max'),
        num_days=('date', 'nunique'),
        avg_volume=('volume', 'mean'),
        last_close=('close', 'last')
    ).reset_index()
    return stocks.sort_values('avg_volume', ascending=False)


def compute_tunindex(df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute a proxy TUNINDEX from capital-weighted daily returns."""
    if df is None:
        df = load_all_data()
    
    # Use capital as weight proxy
    daily = df.groupby('date').agg(
        total_capital=('capital', 'sum'),
        weighted_close=('close', lambda x: np.average(x, weights=df.loc[x.index, 'capital'].clip(lower=1))),
        num_stocks=('code', 'nunique'),
        total_volume=('volume', 'sum')
    ).reset_index()
    daily.sort_values('date', inplace=True)
    daily['return_pct'] = daily['weighted_close'].pct_change() * 100
    
    # Compute index (base 1000)
    daily['tunindex'] = 1000 * (1 + daily['weighted_close'].pct_change().fillna(0)).cumprod()
    return daily


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, SMA to stock data."""
    df = df.copy()
    
    # SMA 20, 50
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    
    # Daily return
    df['return_pct'] = df['close'].pct_change() * 100
    
    # Volatility (20-day)
    df['volatility_20'] = df['return_pct'].rolling(20).std()
    
    return df
