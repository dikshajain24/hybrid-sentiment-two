# src/utils.py
from pathlib import Path
import re

# ROOT = project root
ROOT = Path(__file__).resolve().parents[1]

def data_root() -> Path:
    return ROOT / "data"

def raw_dir() -> Path:
    return data_root() / "raw"

def processed_dir() -> Path:
    return data_root() / "processed"

def list_raw_csvs():
    return sorted(raw_dir().glob("*.csv"))

def safe_read_csv(path):
    import pandas as pd
    return pd.read_csv(path, low_memory=False)

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    # remove urls and html
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"\<[^\>]*\>", " ", s)
    # keep letters, numbers and basic punctuation
    s = re.sub(r"[^a-z0-9\s\!\?\.\,']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
