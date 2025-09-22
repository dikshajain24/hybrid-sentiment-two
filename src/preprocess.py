# src/preprocess.py
from pathlib import Path
import pandas as pd
from src import utils

def main():
    in_path = utils.processed_dir() / "combined_raw.csv"
    if not in_path.exists():
        print("Missing combined_raw.csv — run ingest_merge first.")
        return
    df = pd.read_csv(in_path, low_memory=False)
    df['text'] = df.get('text', "").fillna("").astype(str)
    df['text_clean'] = df['text'].apply(utils.clean_text)
    df['text_len'] = df['text_clean'].str.len()
    out = utils.processed_dir() / "combined_clean.csv"
    df.to_csv(out, index=False)
    print("Saved combined_clean.csv with", len(df), "rows at", out)

if __name__ == "__main__":
    main()
