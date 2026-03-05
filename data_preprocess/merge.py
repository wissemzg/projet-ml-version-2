from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


def read_mixed_csv(path: Path) -> pd.DataFrame:

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        dtype=str,
        skipinitialspace=True,
        encoding="utf-8",
        encoding_errors="ignore",
    )

    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    if "SEANCE" in df.columns:
        df = df[df["SEANCE"].ne("----------")]

    if "SEANCE" in df.columns:
        df["SEANCE"] = pd.to_datetime(
            df["SEANCE"], dayfirst=True, errors="coerce")

    df["source_file"] = path.name
    df["source_folder"] = str(path.parent)

    return df


all_csv_files = sorted(DATA_DIR.rglob("*.csv"))

dfs = []
for p in all_csv_files:
    try:
        dfs.append(read_mixed_csv(p))
    except Exception as e:
        print(f"Failed: {p} -> {e}")


merged_df = pd.concat(dfs, ignore_index=True, sort=False)

print("Files loaded:", len(dfs))
print("Merged shape:", merged_df.shape)
print("Columns:", merged_df.columns.tolist())
merged_df.to_csv("merged_all.csv", index=False, encoding="utf-8")
