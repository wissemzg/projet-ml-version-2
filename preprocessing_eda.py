import pandas as pd
import numpy as np
import os

def convert_numeric_text(series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series containing numeric values stored as text to float64.
    Handles spaces as thousand separators and commas as decimal separators.
    """
    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    
    cleaned = series.astype(str).str.replace(r'\s+', '', regex=True)
    cleaned = cleaned.str.replace(',', '.', regex=False)
    cleaned = cleaned.replace('nan', np.nan)
    
    return pd.to_numeric(cleaned, errors='coerce')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK A: Preprocesses the raw BVMT dataframe.
    """
    print("--- TASK A: STARTING PREPROCESSING ---")
    df = df.copy()
    
    col_mapping = {
        'SEANCE': 'date',
        'GROUPE': 'group',
        'CODE': 'code',
        'VALEUR': 'name',
        'OUVERTURE': 'open',
        'CLOTURE': 'close',
        'PLUS_BAS': 'low',
        'PLUS_HAUT': 'high',
        'QUANTITE_NEGOCIEE': 'volume',
        'NB_TRANSACTION': 'trades',
        'CAPITAUX': 'turnover'
    }
    df = df.rename(columns=col_mapping)
    print(f"[1] Columns standardized: {list(df.columns)}")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        print("[2] 'date' column parsed as datetime.")
    
    numeric_cols = ['open', 'close', 'low', 'high', 'volume', 'trades', 'turnover']
    for col in numeric_cols:
        if col in df.columns:
            orig_valid_count = df[col].notna().sum()
            df[col] = convert_numeric_text(df[col])
            new_valid_count = df[col].notna().sum()
            
            if orig_valid_count > 0:
                fails = orig_valid_count - new_valid_count
                fail_rate = (fails / orig_valid_count) * 100
            else:
                fail_rate = 0.0
            print(f"[3] Column '{col}' converted. Failure rate: {fail_rate:.2f}%")

    initial_len = len(df)
    subset_drop = [c for c in ['date', 'code', 'close'] if c in df.columns]
    if subset_drop:
        df = df.dropna(subset=subset_drop)
        print(f"[4] Dropped {initial_len - len(df)} rows missing {subset_drop}.")
    
    for col in numeric_cols:
        if col in df.columns and col != 'close':
            missing_count = df[col].isna().sum()
            print(f"    Missingness in '{col}' to be kept: {missing_count} rows.")
    
    if 'code' in df.columns and 'date' in df.columns:
        initial_len = len(df)
        df = df.sort_values(by=['code', 'date'])
        df = df.drop_duplicates(subset=['code', 'date'], keep='last')
        print(f"[5] Dropped {initial_len - len(df)} duplicate rows by (code, date).")
        df = df.sort_values(by=['code', 'date']).reset_index(drop=True)
    
    print("[6] Validating OHLC consistency (low <= open/close <= high)...")
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        cond_low = (df['low'] <= df['open']) & (df['low'] <= df['close'])
        cond_high = (df['high'] >= df['open']) & (df['high'] >= df['close'])
        
        valid_ohlc = df[['open', 'high', 'low', 'close']].notna().all(axis=1)
        violations = valid_ohlc & ~(cond_low & cond_high)
        
        num_violations = violations.sum()
        if num_violations > 0:
            print(f"    WARNING: Found {num_violations} OHLC violations.")
            ohlc_issues = df[violations].copy()
        else:
            print("    No OHLC violations found.")
            ohlc_issues = pd.DataFrame()
            
    os.makedirs('data', exist_ok=True)
    parquet_path = 'data/cleaned_bvmt.parquet'
    csv_path = 'data/cleaned_bvmt.csv'
    
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[8] Saved cleaned data to {parquet_path}")
    except Exception as e:
        print(f"[8] Failed to save parquet (pyarrow missing?): {e}")
        
    df.to_csv(csv_path, index=False)
    print(f"[8] Saved cleaned data to {csv_path}")
    
    print("--- TASK A: COMPLETE ---\n")
    return df


def data_understanding(df: pd.DataFrame):
    """
    TASK B: Performs EDA and outputs summary tables and a markdown report.
    """
    print("--- TASK B: STARTING DATA UNDERSTANDING ---")
    
    shape = df.shape
    features = list(df.columns)
    
    date_min, date_max, unique_sessions = None, None, 0
    if 'date' in df.columns:
        date_min = df['date'].min()
        date_max = df['date'].max()
        unique_sessions = df['date'].nunique()
    
    unique_stocks, top_10_stocks = 0, {}
    if 'code' in df.columns:
        unique_stocks = df['code'].nunique()
        top_10_stocks = df['code'].value_counts().head(10).to_dict()
    
    col_stats = []
    for col in df.columns:
        pct_missing = df[col].isna().mean() * 100
        n_unique = df[col].nunique()
        examples = df[col].dropna().unique()[:3].tolist()
        col_stats.append({
            'column': col,
            'dtype': str(df[col].dtype),
            '%_missing': round(pct_missing, 2),
            'n_unique': n_unique,
            'examples': examples
        })
    eda_summary_table = pd.DataFrame(col_stats)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_stats = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) > 0:
            numeric_stats.append({
                'column': col,
                'min': s.min(),
                'max': s.max(),
                'mean': s.mean(),
                'std': s.std(),
                'median': s.median(),
                '5th_pct': s.quantile(0.05),
                '95th_pct': s.quantile(0.95)
            })
    numeric_stats_table = pd.DataFrame(numeric_stats) if numeric_stats else pd.DataFrame()
    
    missingness_counts = df.isna().sum()
    
    negatives = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'trades', 'turnover']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            negatives[col] = neg_count
            
    print("\n[EDA Summary Table]")
    print(eda_summary_table.to_string())
    
    print("\n[Numeric Stats Table]")
    print(numeric_stats_table.to_string())
    
    print("\n[Negative Values Detection]")
    for k, v in negatives.items():
        print(f"  {k}: {v} negative values")
            
    md_content = f"""# BVMT Data Understanding Report

## 1. Dataset Overview
- **Shape**: {shape[0]} rows, {shape[1]} columns
- **Features**: {', '.join(features)}
- **Date Range**: {date_min.strftime('%Y-%m-%d') if pd.notnull(date_min) else 'N/A'} to {date_max.strftime('%Y-%m-%d') if pd.notnull(date_max) else 'N/A'}
- **Unique Trading Sessions**: {unique_sessions}
- **Unique Stocks (Codes)**: {unique_stocks}

## 2. Top 10 Stocks by Record Count
"""
    for code, count in top_10_stocks.items():
        md_content += f"- **{code}**: {count} records\n"
        
    md_content += "\n## 3. Data Quality and Sanity Checks\n"
    md_content += "### Missingness Summary (Counts)\n"
    for col, count in missingness_counts.items():
        if count > 0:
            md_content += f"- **{col}**: {count} missing values\n"
        
    md_content += "\n### Negative Values Detection\n"
    for col, count in negatives.items():
        if count > 0:
            md_content += f"- **{col}**: {count} negative occurrences\n"
        
    md_content += "\n*(See Python console output for detailed column data and summary statistics.)*\n"

    with open('data_understanding.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("\n--- TASK B: COMPLETE (Report saved to 'data_understanding.md') ---")
    
    return eda_summary_table, numeric_stats_table


if __name__ == "__main__":
    from pathlib import Path
    
    csv_path = Path("data/merged_csv/cleaned_file.csv")
    if csv_path.exists():
        print(f"Loading data from {csv_path}...")
        df_demo = pd.read_csv(csv_path)
        df_demo.columns = df_demo.columns.str.strip()
        
        # Produce the standard text output and markdown report
        data_understanding(df_demo)
        
        try:
            from data_preprocess.pdf_report import generate_pdf_report
            print("Generating PDF report...")
            generate_pdf_report(df_demo)
        except ImportError as e:
            print(f"Failed to load PDF generation module: {e}")
    else:
        print("Import these functions into your notebook/script or run with your DataFrame.")
