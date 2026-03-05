import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_PATH = Path(r"./data")
CONVERTED_DIR = BASE_PATH / "_converted_csv"
CONVERTED_DIR.mkdir(exist_ok=True)


def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            raw[:5000].decode(enc)
            return enc
        except:
            pass
    return "latin-1"


def find_header_and_sep(text_lines):
    """
    Find the header line (contains SEANCE) and infer separator.
    Returns (skiprows, sep, use_fwf)
    """
    header_idx = None
    for i, line in enumerate(text_lines[:50]):
        if "SEANCE" in line.upper() and "CODE" in line.upper():
            header_idx = i
            header_line = line
            break

    if header_idx is None:
        header_idx = 0
        header_line = text_lines[0] if text_lines else ""

    if "\t" in header_line:
        return header_idx, "\t", False
    if ";" in header_line:
        return header_idx, ";", False
    if "," in header_line:
        return header_idx, ",", False

    return header_idx, r"\s+", False


def read_bvmt_txt(path: Path) -> pd.DataFrame:
    enc = detect_encoding(path)
    with open(path, "r", encoding=enc, errors="replace") as f:
        lines = [next(f, "") for _ in range(50)]

    skiprows, sep, _ = find_header_and_sep(lines)

    try:
        df = pd.read_csv(
            path,
            encoding=enc,
            sep=sep,
            engine="python",
            dtype=str,
            skiprows=skiprows,
            on_bad_lines="skip"
        )

        if df.shape[1] <= 1:
            raise ValueError("Parsed as 1 column; fallback to FWF")
        return df
    except:
        pass

    df = pd.read_fwf(
        path,
        encoding=enc,
        dtype=str,
        skiprows=skiprows
    )
    return df


txt_files = sorted(BASE_PATH.rglob("*.txt"))
print(f"TXT files found: {len(txt_files)}")

failed_txt = []
converted_paths = []

for f in txt_files:
    try:
        df = read_bvmt_txt(f)
        out_path = CONVERTED_DIR / (f.stem + ".csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        converted_paths.append(out_path)
        print(
            f"✅ Converted: {f.name} -> {out_path.name}  (cols={df.shape[1]}, rows={df.shape[0]})")
    except Exception as e:
        failed_txt.append((f.name, str(e)))
        print(f"❌ Failed: {f.name} -> {e}")

if failed_txt:
    print("\nFailed TXT conversions:")
    for name, msg in failed_txt:
        print("-", name, "->", msg)
