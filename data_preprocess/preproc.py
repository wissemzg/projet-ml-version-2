import pandas as pd
import warnings
import re
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_PATH = Path(r"./data")
EXTRA_CSV_DIR = BASE_PATH / "_converted_csv"
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)
csv_files = sorted(set(BASE_PATH.rglob("*.csv")) |
                   set(EXTRA_CSV_DIR.rglob("*.csv")))
print(f"CSV files found: {len(csv_files)}")
for i, f in enumerate(csv_files, 1):
    print(f"{i:>3}. {f.name}")


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except:
            pass
    raise ValueError(f"Could not read {path}")


def normalize_columns(cols):
    out = []
    for c in cols:
        c = str(c).strip().lower()
        c = re.sub(r"[^\w]+", "_", c)
        c = re.sub(r"__+", "_", c).strip("_")
        out.append(c)
    return out


COLUMN_MAP = {
    "seance": "date",
    "groupe": "group",
    "code": "isin",
    "valeur": "name",
    "ouverture": "open",
    "cloture": "close",
    "plus_bas": "low",
    "plus_haut": "high",
    "quantite_negociee": "volume",
    "nb_transaction": "num_trades",
    "capitaux": "value_traded",
}


def coerce_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(
        "\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def standardize_bvmt(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = normalize_columns(df.columns)

    df = df.rename(columns={c: COLUMN_MAP[c]
                   for c in df.columns if c in COLUMN_MAP})

    
    df["source_file"] = source_file

    
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    
    if "date" not in df.columns or "isin" not in df.columns:
        return pd.DataFrame()

    
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    for c in ["open", "high", "low", "close", "volume", "num_trades", "value_traded"]:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    
    df = df.dropna(subset=["date", "isin"])
    df = df[df["isin"].astype(str).str.len() > 0]

    
    df = df.sort_values(["isin", "date"]).drop_duplicates(
        ["isin", "date"], keep="last")

    return df



parts, failed = [], []

for f in csv_files:
    try:
        raw = read_csv_robust(f)
        clean = standardize_bvmt(raw, f.name)
        if len(clean):
            parts.append(clean)
    except Exception as e:
        failed.append((f.name, str(e)))

bvmt_dataset = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

print("\n=== Unified Dataset Summary ===")
print("Rows     :", len(bvmt_dataset))
print("Columns  :", list(bvmt_dataset.columns))
if len(bvmt_dataset):
    print("ISINs    :", bvmt_dataset["isin"].nunique())
    print("Date range:", bvmt_dataset["date"].min(
    ), "->", bvmt_dataset["date"].max())

if failed:
    print("\nFailed files (first 10):")
    for name, msg in failed[:10]:
        print("-", name, "->", msg)


out_csv = OUT_DIR / "bvmt_unified.csv"
bvmt_dataset.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\n✅ Saved unified dataset: {out_csv}")


try:
    out_parquet = OUT_DIR / "bvmt_unified.parquet"
    bvmt_dataset.to_parquet(out_parquet, index=False)
    print(f"✅ Saved parquet: {out_parquet}")
except:
    print("ℹ️ Parquet not saved (install pyarrow: pip install pyarrow)")
