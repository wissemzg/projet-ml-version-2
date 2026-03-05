import pandas as pd
import csv
from pathlib import Path
from io import StringIO

def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:6000]
    # UTF-16 often has BOM or many null bytes
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff") or b"\x00" in raw[:200]:
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            raw.decode(enc)
            return enc
        except:
            pass
    return "latin-1"

def looks_like_html(text: str) -> bool:
    t = (text or "").lstrip().lower()
    return t.startswith("<!doctype") or t.startswith("<html") or ("<table" in t)

def read_csv_fix(path: Path) -> pd.DataFrame:
    """
    Fix for: Could not read data\\histo_cotation_2023.csv etc.
    - tries encoding (utf-16/utf-8/latin-1)
    - tries separators (;, tab, , , |, whitespace)
    - if file is actually HTML (web_*), parses table via read_html
    """
    enc = detect_encoding(path)

    # read a small preview for HTML detection + separator detection
    with open(path, "r", encoding=enc, errors="replace") as f:
        preview = f.read(3000)

    # HTML disguised as CSV (common for web exports)
    if looks_like_html(preview):
        tables = pd.read_html(StringIO(preview))
        if not tables:
            raise ValueError("HTML detected but no table found.")
        df = max(tables, key=lambda t: t.shape[0] * t.shape[1]).astype(str)
        return df

    # Try multiple separators (your 2023+ needs ';')
    seps = [";", "\t", ",", "|", r"\s+"]

    last_err = None
    for sep in seps:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=sep,
                engine="python",         # needed for regex sep
                dtype=str,
                on_bad_lines="skip"
            )
            if df.shape[1] > 1:  # success
                return df
        except Exception as e:
            last_err = e

        # last resort: broken quotes
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                sep=sep,
                engine="python",
                dtype=str,
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
                escapechar="\\"
            )
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e

    raise ValueError(f"Could not read {path} (encoding={enc}). Last error: {last_err}")
raw = read_csv_robust(f)
raw = read_csv_fix(f)